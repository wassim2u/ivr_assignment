from sympy import *
init_printing(use_unicode=True)
alpha, a, d, theta = symbols('alpha a d theta')


def rot_alpha_aroundx(alpha_val):
    M = Matrix([
        [1, 0, 0, 0],
        [0, cos(alpha), -sin(alpha),0],
        [0, sin(alpha), cos(alpha), 0],
        [0,0,0,1]
    ])
    M = M.subs(alpha,alpha_val)
    return M


def displacement_xaxis(a_value):
    M = Matrix([
        [1,0,0,a],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
    M= M.subs(a,a_value)
    return M

def displacement_zaxis(d_value):
    M = Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]
    ])
    M= M.subs(d,d_value)
    return M

def rot_theta_aroundz(theta_value):
    M = Matrix([
        [cos(theta), -sin(theta), 0, 0],
        [sin(theta), cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    M = M.subs(theta,theta_value)
    return M


#eg. A01, A12, etc..
def calculate_for_current_frame(theta_value, d_value, a_value, alpha_value):
    M = rot_theta_aroundz(theta_value) * displacement_zaxis(d_value) * displacement_xaxis(a_value) * rot_alpha_aroundx(alpha_value)
    return M



def print_matrix_nicely(matrix):
    pprint(matrix)

def print_in_latex(matrix):
    print(latex(matrix))


def set_frame01():
    # Type symbols and values you might need
    # frame01
    theta1 = symbols(" theta1 ")
    theta1 = theta1 -pi / 2
    d1 = 2.5
    a1 = 0
    alpha1 = pi / 2
    frame01 = calculate_for_current_frame(theta1, d1, a1, alpha1)
    return frame01

def set_frame12():
    ###frame12##
    theta2 = symbols(" theta2 ")
    # change these values
    theta2 = pi / 2 + theta2
    d2 = 0
    a2 = 0
    alpha2 = pi / 2
    #
    frame12 = calculate_for_current_frame(theta2, d2, a2, alpha2)
    return frame12

def set_frame23():
    ###frame23###
    theta3 = symbols(" theta3 ")
    # change these values
    theta3 =  theta3
    d3 = 0
    a3 = 3.5
    alpha3 = - pi / 2
    #
    frame23 = calculate_for_current_frame(theta3, d3, a3, alpha3)
    return  frame23


def set_frame34():
    ###frame34###
    theta4 = symbols(" theta4 ")
    # change these values
    theta4 = theta4
    d4 = 0
    a4 = 3
    alpha4 = 0
    #
    frame34 = calculate_for_current_frame(theta4, d4, a4, alpha4)
    return frame34

if __name__ == "__main__":

    frame01 = set_frame01()
    frame12 = set_frame12()

    frame23 = set_frame23()
    frame34 = set_frame34()

    ###frame02###
    frame02 = frame01 * frame12

    ###frame03###
    frame03 = frame02*frame23

    ###frame04###
    frame04 = frame03*frame34

    ##frame13###
    frame13 = frame12*frame23

    #print stuff either using print or pprint
    pprint(frame01[:,3])
    pprint(frame12[:,3])
    pprint(frame34[:,3])

    #testing
    M = frame04[:,3]
    theta2 = symbols(" theta2 ")
    theta3 = symbols(" theta3 ")
    theta4 = symbols(" theta4 ")

    pprint(M.subs(theta2,pi/2).subs(theta3,0).subs(theta4,0))

    M = Matrix([
        [cos(theta), 0, sin(theta), 0],
        [0, 1, 0, 0],
        [-sin(theta), 0, cos(theta), 0],
        [0, 0, 0, 1]
    ])
    M =M.subs(theta,-0.17)

    pprint(M*Matrix([4.13595421, 0.68221925, 3.45373496,1]))