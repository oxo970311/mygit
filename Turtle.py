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
