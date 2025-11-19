from jetracer.nvidia_racecar import NvidiaRacecar
import time

# JetRacer 객체 생성
car = NvidiaRacecar()

print("=== JetRacer Servo & Motor Combined Test ===")

# ------------------------
# 🧭 서보모터 테스트 (좌 ↔ 우 ↔ 중앙)
# ------------------------
print("Steering test...")

car.steering = 0.0
time.sleep(1)

# 왼쪽 끝
car.steering = 1.0
time.sleep(1)

# 오른쪽 끝
car.steering = -1.0
time.sleep(1)

# 중앙
car.steering = 0.0
time.sleep(1)

print("Steering test complete!")

# ------------------------
# ⚡ 모터 테스트 (전진 ↔ 정지 ↔ 후진 ↔ 정지)
# ------------------------
# print("Motor test...")

# # 전진
# car.throttle = -0.4
# print("Forward...")
# time.sleep(2)

# # 정지
# car.throttle = 0.0
# print("Stop.")
# time.sleep(1)

# # 후진
# car.throttle = 0.3
# print("Backward...")
# time.sleep(2)

# # 정지
# car.throttle = 0.0
# print("Stop.")

print("=== All tests complete ===")

