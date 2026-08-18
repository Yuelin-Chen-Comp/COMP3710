import turtle

# ==========================================
# 🎛️ 你可以在这里调整“模拟器”的参数！(Adjust Parameters Here)
# ==========================================
BRANCH_ANGLE = 30      # 分支角度 (Angle)：改变树枝分叉的宽窄
LENGTH_SCALE = 0.7     # 长度缩放率 (Length Scale)：决定下一级树枝变短的程度
MIN_LENGTH = 20         # 最小长度 (递归终止条件)：决定树的“深度 (Depth)”

def draw_branch(t, branch_length):
    """
    这是一个递归函数，用于绘制分形树的树枝。
    (This is a recursive function to draw the branches of the fractal tree.)
    """
    # 只要树枝长度大于最小长度，就继续生长
    if branch_length > MIN_LENGTH:
        # 1. 向前画出当前的树干
        t.forward(branch_length)
        
        # 2. 向右转，准备画右边的分叉
        t.right(BRANCH_ANGLE)
        draw_branch(t, branch_length * LENGTH_SCALE) # 【递归调用】画右树枝
        
        # 3. 向左转，准备画左边的分叉 (注意要转两个角度才能回到左边)
        t.left(BRANCH_ANGLE * 2)
        draw_branch(t, branch_length * LENGTH_SCALE) # 【递归调用】画左树枝
        
        # 4. 转回原来的方向，并退回原点，以便上一级树枝继续绘制
        t.right(BRANCH_ANGLE)
        t.backward(branch_length)

def main():
    # 设置画布和画笔
    screen = turtle.Screen()
    screen.title("COMP3710 Fractal Tree Simulator")
    
    t = turtle.Turtle()
    t.speed("fastest") # 设置最快绘画速度
    t.color("forest green") # 设置画笔颜色
    t.left(90) # 让画笔初始方向朝上
    t.up()
    t.backward(150) # 把起点往下移一点
    t.down()

    # 开始绘制第一根主干，初始长度设定为 100
    draw_branch(t, 100)

    # 点击窗口退出
    screen.exitonclick()

if __name__ == "__main__":
    main()
