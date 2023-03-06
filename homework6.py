import pandas
import numpy
import matplotlib.pyplot as plt

df = pandas.read_csv("HW6_prob1_dataset1.csv")
df = df.to_numpy()
#Ok, now we have all the points, and we just need to create the A matrix.
shape = df.shape
Amatrix = []
for k in range(shape[0]):
    Amatrix.append([])

b = []
#Create an array A.
index = 0
for input_row in df:
    b.append(-1)
    row = Amatrix[index]
    row.append(input_row[0]*input_row[0])
    row.append(input_row[0]*input_row[1])
    row.append(input_row[1]*input_row[1])
    row.append(input_row[0])
    row.append(input_row[1])
    index +=1
print(len(b))
b_mat = numpy.asarray(b)
important_A = numpy.asarray(Amatrix)
A_transpose = important_A.transpose()

ata = numpy.matmul(A_transpose, important_A)
atb = numpy.matmul(A_transpose, b_mat)

c = numpy.linalg.solve(ata, atb)
print(c)
#BEGIN PLOTTING NONSENSE
#delta = 0.025
#x = numpy.arange(-5.0, 5.0, delta) #Ranges from x1 -- x2, incrementing by delta
#y = numpy.arange(-5.0, 5.0, delta)


x = numpy.linspace(-20, 20, 400)
y = numpy.linspace(-20, 20, 400)
x, y = numpy.meshgrid(x, y)
#X, Y = numpy.meshgrid(x, y) #meshgrid returns coordinate vectors?
#Z1 = numpy.exp(-X**2 - Y**2)
#Z2 = numpy.exp(-(X - 1)**2 - (Y - 1)**2)
#Z = (Z1 - Z2) * 2
plt.scatter(df[:,0], df[:,1])
#plt.axhline(0, alpha=.1) #Plots axes, with a thickness of 0.1
#plt.axvline(0, alpha=.1)
#plt.axis('scaled')
plt.xlim(right = 15)
plt.xlim(left = -5)
plt.ylim(top = 15)
plt.ylim(bottom = -15)
z = (c[0]*(x**2) + c[1]*x*y + c[2]*(y**2) + c[3] * x + c[4] * y+1)
plt.contour(x, y, z, [0], colors='k')
plt.show()










