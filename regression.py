import numpy as np
import matplotlib.pyplot as plt
import math

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)
    print("X mean: ",m_x , "Y Mean: ",m_y)
    # calculating cross-deviation and deviation about x
    '''
    SS_xy = np.sum(y*x) - n* m_y* m_x
    SS_xx = np.sum(x*x) - n* m_x* m_x
    '''
    num=0
    den=0
    for i in range(n):
         num = num + (x[i] - m_x) * (y[i] - m_y)
         den = den + math.pow(x[i] - m_x, 2) 
    b1 = num/den
    b0 = m_y - (b1 * m_x)
    print(b1, b0)
    '''
     # calculating regression coefficients
    b1 = SS_xy / SS_xx
    b0 = m_y - b1*m_x
    '''
    return(b0, b1)

def plot_regression_line(x, y, b,x_t=0,y_pred_val=0):
    #x, y are arrays. b= coeff, 
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "b", marker = "o", s = 30)
    # predicted response vector
    y_pred = b[0] + b[1]*x
    # plotting the regression line
    plt.plot(x, y_pred, color = "g")
    if x_t!=0 and y_pred_val!=0:    #for predicting
        plt.plot(x_t, y_pred_val,"ro")
    # putting labels
    plt.xlabel('X')
    plt.ylabel('Y')
    # function to show plot
    plt.show()

def main():
    # observations
    '''
    x = np.array([6, 9, 2, 15, 10, 16, 11, 16])
    y = np.array([95, 80, 10, 50, 45, 98, 38, 83])
   '''
    x = np.array([1, 2, 4, 3, 5])
    y = np.array([3, 4, 2, 4, 5])
    '''
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    # estimating coefficients
    '''
    print("X[]:",x,"\nY[]:",y)
    b = estimate_coef(x, y)
    #print("b0: ",b[0])
    print("Estimated coefficients:\n b_0 = {}  \n b_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plot_regression_line(x, y, b)

    #predict value for X=8
    print("Enter X value : ")
    x_t = int(input())
#    x_t = 8
    y_pred = b[0]+(b[1]*x_t)
    print("Predicted value for x = {} is y = {} ",x_t,y_pred)

    #Final Plot
    plot_regression_line(x, y, b,x_t,y_pred)

if __name__ == "__main__":
    main()
