import numpy as np


def skew(v):
    return np.array([[0, -v[2,0], v[1,0]], 
                     [v[2,0], 0, -v[0,0]], 
                     [-v[1,0], v[0,0], 0]])


def Sampson(p1, p2, F):
    num = (p1.T.dot(F).dot(p2))[0,0] ** 2
    a = p1.T.dot(F)
    b = F.dot(p2)
    den = a[0,0]**2 + a[0,1]**2 + b[0,0]**2 + b[1,0]**2
    print(num, den)
    return num/den


R_ec0 = np.array([[-5.89144348e-07,-1.00000016e+00,4.92948109e-09],
                  [ 1.57108457e-02,-4.32708253e-09,9.99876577e-01],
                  [-9.99876739e-01, 5.89149081e-07,1.57108457e-02]])
R_ec1 = np.array([[ 0.49374182,-0.86959812,-0.0042624],
                  [ 0.01250955, 0.00220151, 0.99991933],
                  [-0.86951859,-0.49375531, 0.01196526]])
T_c0e = np.array([[239.57553101],[ 30.0000019 ],[ -0.54384458]])
T_c1e = np.array([[239.26230621],[ 20.17891073],[ -0.5387274 ]])
pi0 = np.array([[314], [230], [1]])
pi01 = np.array([[422], [234], [1]])
pi1 = np.array([[109], [228], [1]])
pi11 = np.array([[235], [234], [1]])
f, u0, v0 = 320, 320, 240
K = np.array([[f,0,u0], [0,f,v0], [0,0,1]])

R_c1c0 = R_ec0.dot(R_ec1.T)
T_c1c0 = R_ec0.dot(T_c1e - T_c0e)
F = np.dot(skew(T_c1c0), R_c1c0)
print( Sampson(np.linalg.inv(K).dot(pi0), np.linalg.inv(K).dot(pi1), F) )