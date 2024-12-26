from math import cos, sin, tan, atan2, pi, atan
import matplotlib.pyplot as plt

x = [0]
y = [0]
v = [0.1]
phi = [0]
b = [0]
t = 0

lr = 1
lf = 1

for i in range(10000):
    # control input
    a = 0
    steering = 1 * pi/180

    # delta t = 1
    x.append(x[t] + v[t] * cos(phi[t]+b[t]))
    y.append(y[t] + v[t] * sin(phi[t]+b[t]))
    v.append(v[t] + a)
    phi.append(phi[t] + v[t]/lr * sin(b[t]))
    b.append(b[t] + atan2(lr*tan(steering), (lr+lf)))


    print(f't: {t} x: {x[t]} y: {y[t]} v: {v[t]} phi: {phi[t]} b: {b[t]}')

    t += 1

# plot

fontsize = 20
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(x, y)

plt.xlabel('x', fontsize=fontsize)
plt.ylabel('y', fontsize=fontsize)

plt.tight_layout()
plt.show()

