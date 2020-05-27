def detect():
    while True:
        res = 0
        str_grad = input()
        grads = str_grad.split(" ")
        for i in range(len(grads)):
            grad = grads[i]
            res += int(grad[3:])
        print(res)

detect()