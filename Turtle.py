# import turtle
# import random
#
# t = turtle.Turtle()
# t.shape("turtle")
# t_s = t.clone()
# t.color("green")
# t_s.color("orange")
# # t.speed(1)
# # t_s.speed(1)
#
# t.penup()
# t.goto(300,60)
# t.pendown()
# t.circle(40)
# t.penup()
# t.setpos(-300, 100)
# t.pendown()
#
# t_s.penup()
# t_s.goto(300,-140)
# t_s.pendown()
# t_s.circle(40)
# t_s.penup()
# t_s.setpos(-300, -100)
# t_s.pendown()
#
#
# for _ in range(100):
#     r_number1 = random.randint(1, 100)
#     r_number2 = random.randint(1, 100)
#     t.penup(); t_s.penup()
#     t.fd(r_number1)
#     t_s.fd(r_number2)
#     t.pendown(); t_s.pendown()
#     if t.pos() >= (300,60):
#         print("green turtle win !")
#         break
#
#     elif t_s.pos() >= (300,-140):
#         print("orange turtle win !")
#         break

import turtle
import random
import time

# 기본 터틀 설정
t = turtle.Turtle()
t.shape("turtle")

# 줄의 중앙선 그리기
t.penup()
t.goto(0, -50)
t.setheading(90)
t.pendown()
t.forward(120)

# 왼쪽 팀 클론
left_team = []
x1 = -100
y1 = 20
for i in range(5):
    ti = t.clone()
    ti.color("blue")
    ti.penup()
    ti.setheading(180)
    ti.forward(100)
    ti.goto(x1, y1)
    ti.pendown()
    left_team.append(ti)
    x1 -= 40

# 오른쪽 팀 클론
right_team = []
x2 = 100
y2 = 20
for j in range(5):
    tj = t.clone()
    tj.color("red")
    tj.penup()
    tj.setheading(0)
    tj.forward(100)
    tj.goto(x2, y2)
    tj.pendown()
    right_team.append(tj)
    x2 += 40

t.hideturtle()

# 줄다리기 시뮬레이션
center_x = 0  # 줄의 중심

while abs(center_x) < 200:
    pull = random.randint(-100, 100)
    center_x += pull

    for ti in left_team:
            ti.penup()
            ti.setx(ti.xcor() + pull)
            ti.pendown()
    for tj in right_team:
            tj.penup()
            tj.setx(tj.xcor() + pull)
            tj.pendown()

    time.sleep(0.2)

# 승리 메시지
msg = turtle.Turtle()
msg.penup()
msg.hideturtle()
msg.goto(0, -100)
msg.write("Left Team Wins!" if center_x < 0 else "Right Team Wins!", align="center", font=("Arial", 16, "bold"))

turtle.done()
