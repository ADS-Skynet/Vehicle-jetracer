import argparse

from dcas_client import DCASMockInput, DCASPolicyClient


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"FAIL: {message}")


def run_case(client: DCASPolicyClient, name: str, mock_input: DCASMockInput) -> dict:
    outputs = client.evaluate(mock_input)
    expect(len(outputs) == mock_input.ticks, f"{name}: expected {mock_input.ticks} output rows, got {len(outputs)}")
    output = outputs[-1]
    print(f"[PASS] {name}: {output['step_b_next_state']}/{output['hmi_action']}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DCAS subprocess policy scenarios")
    parser.add_argument("--runner", default=None, help="Override path to dcas_policy_runner")
    parser.add_argument("--auto-build", action="store_true", help="Auto-build C++ runner if missing")
    args = parser.parse_args()

    client = DCASPolicyClient(runner_path=args.runner)
    client.preflight(auto_build=args.auto_build)

    print("\n=== Step B: Attentive Snapshot & Critical Reason ===\n")

    # S-B-002: is_attentive=true but critical reason forces ABSENT
    attentive_with_critical = run_case(
        client,
        "S-B-002_attentive_with_critical_reason",
        DCASMockInput(attentive=True, reason="intoxicated", delta_s=1.0),
    )
    expect(attentive_with_critical["step_b_next_state"] == "OK", "attentive snapshot should stay OK")
    expect(attentive_with_critical["reason"] == "none", "attentive snapshot should normalize reason to none")
    expect(
        not attentive_with_critical["reengagement_confirmed_200ms"],
        "OK state should not emit reengagement confirmation",
    )

    # S-B-001: reason_ts_ms mismatch -> snapshot validation blocks transition
    # (Even critical reason won't trigger if timestamps don't match - safety against async VLM)
    mismatch_hold = run_case(
        client,
        "S-B-001_timestamp_mismatch_blocks_transition",
        DCASMockInput(
            attentive=False,
            reason="drowsy",
            timestamp_ms=1000,
            reason_timestamp_ms=1001,
            delta_s=2.5,
            jetracer_input_0_4=0.2,
        ),
    )
    expect(mismatch_hold["step_b_next_state"] == "OK", "timestamp mismatch should block transition (hold state)")
    expect(mismatch_hold["hmi_action"] == "INFO", "held OK state remains INFO")

    print("\n=== Step B: State Transitions ===\n")

    # S-B-101: OK -> WARNING (MID band, 2s inattentive)
    warning = run_case(
        client,
        "S-B-101_mid_band_warning_drowsy",
        DCASMockInput(attentive=False, reason="drowsy", delta_s=2.5, jetracer_input_0_4=0.2),
    )
    expect(warning["step_b_next_state"] == "WARNING", "drowsy 2.5s should enter WARNING in MID band")
    expect(warning["hmi_action"] == "EOR", "WARNING should map to EOR")

    # S-B-102: WARNING -> ESCALATION (MID band, 4s inattentive)
    escalation = run_case(
        client,
        "S-B-102_mid_band_escalation_phone",
        DCASMockInput(attentive=False, reason="phone", delta_s=1.0, ticks=4, jetracer_input_0_4=0.2),
    )
    expect(escalation["step_b_next_state"] == "ESCALATION", "4s MID-band inattentive should reach ESCALATION")
    expect(escalation["hmi_action"] == "DCA", "ESCALATION should map to DCA")

    # S-B-103: ESCALATION -> ABSENT (MID band, 8s inattentive)
    absent_by_timer = run_case(
        client,
        "S-B-103_mid_band_absent_drowsy",
        DCASMockInput(attentive=False, reason="drowsy", delta_s=1.0, ticks=8, jetracer_input_0_4=0.2),
    )
    expect(absent_by_timer["step_b_next_state"] == "ABSENT", "8s MID-band inattentive should reach ABSENT")
    expect(absent_by_timer["mrm_active"], "ABSENT should activate MRM")
    expect(absent_by_timer["throttle_limit"] == 0.0, "ABSENT should zero throttle")

    # S-B-104: critical reason -> immediate ABSENT
    critical = run_case(
        client,
        "S-B-104_critical_unresponsive",
        DCASMockInput(attentive=False, reason="unresponsive", delta_s=0.1, jetracer_input_0_4=0.1),
    )
    expect(critical["step_b_next_state"] == "ABSENT", "critical reason should jump to ABSENT")
    expect(critical["mrm_active"], "ABSENT should activate MRM")

    print("\n=== Step B: No Reason Input ===\n")

    # S-B-003: is_attentive=no but no reason yet (expect ESCALATION by timer)
    no_reason = run_case(
        client,
        "S-B-003_no_reason_timer_escalation",
        DCASMockInput(attentive=False, reason="none", delta_s=1.0, ticks=4, jetracer_input_0_4=0.2),
    )
    expect(no_reason["step_b_next_state"] == "ESCALATION", "4s with no reason should still escalate by timer")
    expect(no_reason["reason"] == "none", "no reason input should remain none")
    expect(no_reason["hmi_action"] == "DCA", "ESCALATION with no reason should map to DCA")

    print("\n=== Step B: Recovery & Reengagement ===\n")

    # S-B-201: WARNING state with recovery signal
    warning_recovery = run_case(
        client,
        "S-B-201_warning_recovery_200ms_eor_mitigation",
        DCASMockInput(
            attentive=False,
            reason="phone",
            delta_s=2.1,  # Triggers WARNING
            jetracer_input_0_4=0.2,
        ),
    )
    expect(warning_recovery["step_b_next_state"] == "WARNING", "2.1s should enter WARNING")
    expect(warning_recovery["hmi_action"] == "EOR", "WARNING should map to EOR")

    # S-B-202: Verify escalation continues if inattention persists
    warning_to_escalation = run_case(
        client,
        "S-B-202_warning_escalation_progression",
        DCASMockInput(
            attentive=False,
            reason="phone",
            delta_s=1.0,
            ticks=4,  # 4 ticks * 1s = 4s total inattention
            jetracer_input_0_4=0.2,
        ),
    )
    expect(warning_to_escalation["step_b_next_state"] == "ESCALATION", "4s should reach ESCALATION")

    print("\n=== Step C: Policy Output ===\n")

    # S-C-002: Verify throttle limit scaling by state
    throttle_ok = run_case(
        client,
        "S-C-002_throttle_ok_100pct",
        DCASMockInput(attentive=True, reason="none", delta_s=0.1, lkas_throttle=0.8),
    )
    expect(throttle_ok["step_b_next_state"] == "OK", "attentive should remain OK")
    expect(throttle_ok["throttle_limit"] > 0.7, "OK should use nearly full throttle (100%)")

    throttle_warning = run_case(
        client,
        "S-C-002_throttle_warning_reduced",
        DCASMockInput(attentive=False, reason="drowsy", delta_s=2.5, lkas_throttle=0.8, jetracer_input_0_4=0.2),
    )
    expect(throttle_warning["step_b_next_state"] == "WARNING", "drowsy should trigger WARNING")
    # WARNING gain is 0.7 per implementation
    expected_limit = 0.8 * 0.7
    expect(abs(throttle_warning["throttle_limit"] - expected_limit) < 0.01,
           f"WARNING should scale throttle ~{expected_limit}, got {throttle_warning['throttle_limit']}")

    throttle_escalation = run_case(
        client,
        "S-C-002_throttle_escalation_conservative",
        DCASMockInput(attentive=False, reason="phone", delta_s=1.0, ticks=4, lkas_throttle=0.5, jetracer_input_0_4=0.2),
    )
    expect(throttle_escalation["step_b_next_state"] == "ESCALATION", "4s should reach ESCALATION")
    # ESCALATION gain is 0.3 per implementation
    expected_limit = 0.5 * 0.3
    expect(abs(throttle_escalation["throttle_limit"] - expected_limit) < 0.01,
           f"ESCALATION should scale throttle ~{expected_limit}, got {throttle_escalation['throttle_limit']}")

    throttle_absent = run_case(
        client,
        "S-C-002_throttle_absent_zero",
        DCASMockInput(attentive=False, reason="intoxicated", delta_s=0.1, lkas_throttle=0.5),
    )
    expect(throttle_absent["step_b_next_state"] == "ABSENT", "critical reason should trigger ABSENT")
    expect(throttle_absent["throttle_limit"] == 0.0, "ABSENT should always zero throttle")

    print("[PASS] All DCAS test scenarios passed!")


if __name__ == "__main__":
    main()
