import pandas
import numpy
import matplotlib.pyplot as plt

def create2dConic(s):
    df = pandas.read_csv(s)
    df = df.to_numpy()
    #Ok, now we have all the points, and we just need to create the A matrix.
    shape = df.shape
    #Initialize a matrix that has at least the correct number of rows, if nothing else.
    Amatrix = []
    for k in range(shape[0]):
        Amatrix.append([])

    b = []
    #Create an array A.
    maxX = -100000
    minX = 100000
    maxY = -100000
    minY = 100000
    index = 0
    for input_row in df:
        if maxX < input_row[0]:
            maxX = input_row[0]
        if maxY < input_row[1]:
            maxY = input_row[1]
        if minX > input_row[0]:
            minX = input_row[0]
        if minY > input_row[1]:
            minY = input_row[1]
        b.append(-1)
        row = Amatrix[index]
        row.append(input_row[0]*input_row[0])
        row.append(input_row[0]*input_row[1])
        row.append(input_row[1]*input_row[1])
        row.append(input_row[0])
        row.append(input_row[1])
        index +=1
    #print(len(b))
    maxX += 2; maxY += 2; minX -= 2; minY -=2 #Update sizing for prettier graphs

    b_mat = numpy.asarray(b)
    important_A = numpy.asarray(Amatrix)
    A_transpose = important_A.transpose()
    ata = numpy.matmul(A_transpose, important_A)
    atb = numpy.matmul(A_transpose, b_mat)
    c = numpy.linalg.solve(ata, atb)
    #Actually finish solving the problem at hand.
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
    plt.xlim(right = maxX)
    plt.xlim(left = minX)
    plt.ylim(top = maxY)
    plt.ylim(bottom = minY)
    z = (c[0]*(x**2) + c[1]*x*y + c[2]*(y**2) + c[3] * x + c[4] * y+1)
    plt.contour(x, y, z, [0], colors='k')
    plt.show()

def main():
    test1 = "HW6_prob1_dataset1.csv"
    test2 = "HW6_prob1_dataset2.csv"
    create2dConic(test1)
    create2dConic(test2)


main()











