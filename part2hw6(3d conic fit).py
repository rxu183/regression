import pandas
import numpy
import matplotlib.pyplot as plt

def create3dConic(s):
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
    maxZ = -100000
    minZ = 100000
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
        if minZ > input_row[2]:
            minZ = input_row[2]
        if maxZ < input_row[2]:
            maxZ = input_row[2]
        b.append(input_row[2])
        row = Amatrix[index]
        row.append(input_row[0]*input_row[0])
        row.append(input_row[0]*input_row[1])
        row.append(input_row[1]*input_row[1])
        row.append(input_row[0])
        row.append(input_row[1])
        row.append(1)
        index +=1
    maxX += 0.5; maxY += 0.5; minX -= 0.5; minY -=0.5 #Update sizing for prettier graphs
    b_mat = numpy.asarray(b)
    important_A = numpy.asarray(Amatrix)
    A_transpose = important_A.transpose()

    ata = numpy.matmul(A_transpose, important_A)
    atb = numpy.matmul(A_transpose, b_mat)
    c = numpy.linalg.solve(ata, atb)
    #Actually finish solving the problem at hand.
    #print(c)
    #BEGIN PLOTTING NONSENSE
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = numpy.linspace(-10, 10, 400)
    y = numpy.linspace(-10, 10, 400)
    x, y = numpy.meshgrid(x, y)
    ax.scatter(df[:,0], df[:,1], df[:,2])
    #Check maximum Z value for funsies.
    #print(minZ, maxZ)
    plt.xlim(right = maxX)
    plt.xlim(left = minX)
    plt.ylim(top = maxY)
    plt.ylim(bottom = minY)
    ax.set_zlim([minZ, maxZ])

    #ax.zlim(bottom = minZ)
    z = (c[0]*(x**2) + c[1]*x*y + c[2]*(y**2) + c[3] * x + c[4] *y + c[5])
    z[z > maxZ] = numpy.nan
    x[x > maxX] = numpy.nan
    y[y > maxY] = numpy.nan
    ax.plot_surface(x, y, z, alpha=0.5)
    plt.show()

def main():
    test1 = "HW6_prob2_dataset.csv"
    create3dConic(test1)
main()