# CI PROJECT
import numpy as np
import random as rd
import math
from matplotlib import pyplot as plt


class portfolio_wealth_optimization:
    def __init__(self, num_of_portfolios, num_of_variables, x_min, x_max, max_expected_return, min_total_expected_return, max_total_portfolio_risk, C, D):
        # num_of_variables : is the number of decision variables in objective function which representing the number of assets to invest in
        self.num_of_variables = num_of_variables
        # num_of_portfolios : is the number of portfolios we aim to choose the best one of them
        self.num_of_portfolios = num_of_portfolios
        # portfolios : is a matrix where each value in it represents a percentage of capital allocated for each asset , number of rows=number of portfolios, number of columns=number of decision variables (it corresponds to positions in PSO)
        self.portfolios = None
        self.fit = None  # array contains the fitness value for each portfolio
        self.x_min = x_min  # x_min is an array that refers to the minimum percentages of capital could be allocated to each asset
        self.x_max = x_max  # x_max is an array that refers to the maximum percentages of capital could be allocated to each asset
        # max_expected_rete34`4111urn : an array that contains the maximum expected return for each asset (the expected return from each asset if 100% of capital invested in it)
        self.max_expected_return = max_expected_return
        # min_total_expected_return : refers to the minimum total expected return for the whole portfolio
        self.min_total_expected_return = min_total_expected_return
        # self.variance=variance  # variance : an array that contains the expected risk for each asset (the expected risk from each asset if 100% of capital invested in it)
        # max_total_portfolio_risk : refers to the maximum total expected risk
        self.max_total_portfolio_risk = max_total_portfolio_risk
        self.C = C  # C : refers to a 1d array contains coefficients of the decision variables in objective function
        self.D = D  # D : refers to a square matrix contains the coefficients of decision variables in the second term i objective function
        self.gbest_fit = 0  # gbest_fit : is the fitness of the global best portfolio reached so far
        # gbest_position : array contains the global best portfolio reached so far over all
        self.gbest_portfolio = None
        # pbest_fit: personal best fitness-> array contains the best fitness reached by each portfolio
        self.pbest_fit = None
        # pbest_portfolio: personal best portfolio-> matrix contains the portfolios with the best fitness reached by each of them
        self.pbest_portfolio = None
        # expected_risk : matrix contains the expected risk for each portfolio (corresponds to velocity in PSO)
        self.expected_risk = None
        self.total_risk = None  # array contains the total risk for each portfolio
        self.gbest_total_risk = 0  # the total risk value for the best portfolio

    def initial_pop(self):
        # this function generate random numbers to make the initial portfolios
        self.portfolios = np.random.random(
            size=(self.num_of_portfolios, self.num_of_variables))
        self.expected_risk = np.random.random(
            size=(self.num_of_portfolios, self.num_of_variables)) 

    # Budget constraint means that summation of random percentages of capital allocated for each asset must be exactly=1
    def Budget_Constraint(self):
        # note that : Asset allocation constraint is satisfied automatically after implemet Budget constraint, Asset allocation constraint means that the percentage of capital allocated to each asset must be between 0 and 1.
        SUM = []
        for i in range(self.num_of_portfolios):
            # to get the summation of random percentages of capital allocated for each asset
            SUM.append(sum(self.portfolios[i]))
        if max(SUM) > 1 or min(SUM) < 0:  # if constraint is not satisfied
            for i in range(self.num_of_portfolios):
                # rescale the weights to satisfy the Budget constraint
                for j in range(self.num_of_variables):
                    self.portfolios[i][j] = self.portfolios[i][j] / SUM[i]

    def Min_max_investment_constraint(self):
        # Minimum and maximum investment constraint: To ensure that the investment in each asset must fall within a certain range set by the investor
        # the technique used to handle this constraint is "Repair Algorithm" technique
        for i in range(self.num_of_portfolios):
            for j in range(self.num_of_variables):
                if self.portfolios[i][j] > self.x_max[j]:
                    # if percentages of capital allocated for asset number i is greater than the maximum limit assigned to it,then assign it with the maximum limit
                    self.portfolios[i][j] = self.x_max[j]
                elif self.portfolios[i][j] < self.x_min[j]:
                    # if percentages of capital allocated for asset number i is less than the minimum limit assigned to it,then assign it with the minimum limit
                    self.portfolios[i][j] = self.x_min[j]

    def Expected_return_constraint(self):
        # Expected return constraint ensures that the expected return of the portfolio isn't less than the minimum total expected return
        for i in range(self.num_of_portfolios):
            Sum = 0
            for j in range(self.num_of_variables):
                Sum += self.portfolios[i][j] * self.max_expected_return[j]
            if Sum < self.min_total_expected_return:
                self.portfolios[i] = self.pbest_portfolio[i]

    def Risk_constraint(self):
        # Risk constraint ensures that the portfolio's risk, as measured by the variance of the portfolio's returns, does not exceed a certain level
        for i in range(self.num_of_portfolios):

            if self.total_risk[i] > self.max_total_portfolio_risk:
                self.portfolios[i] = self.pbest_portfolio[i]

    def Satisfy_all_constraints(self):

        self.Budget_Constraint()
        self.Min_max_investment_constraint()
        self.Budget_Constraint()
        self.Expected_return_constraint()
        self.Budget_Constraint()
        self.Risk_constraint()
        self.Budget_Constraint()

    def calculate_expected_return(self):
        # this function calculates the fitness value(the expected return value) for each portfolio by applying the objective function using values of variables and coefficients
        #  quadratic programming objective function : C*X + X^(transposed)*D*X
        fitness = []
        for i in range(self.num_of_portfolios):
            X = self.portfolios[i]
            X_transposed = np .transpose(X)
            fitness.append(np.dot(self.C, X) +
                           np.dot(np.dot(X_transposed, self.D), X))
        self.fit = fitness

    def calculate_risk(self):
        # calculate the total risk of the portfolio using variance equation
        avg = []
        variance = np.zeros_like(self.portfolios)
        risk = []
        for i in range(self.num_of_portfolios):
            # calculate the avarage of each portfolio
            avg.append(sum(self.portfolios[i])/self.num_of_variables)
        for i in range(self.num_of_portfolios):
            for j in range(self.num_of_variables):
                # calculate (x-avg)^2
                variance[i][j] = (self.portfolios[i][j]-avg[i])**2
        for i in range(self.num_of_portfolios):
            # claculate the variance for each portfolio
            risk.append(    sum(variance[i])   /  (self.num_of_variables-1)  )
        self.total_risk = risk

    def update_gbest(self):
        # this function updates the global best portfolio reached so far
        max_fit = max(self.fit)
        max_fit_indx = np.argmax(self.fit, 0, None) # indix of max
        if max_fit > self.gbest_fit:
            self.gbest_portfolio = self.portfolios[max_fit_indx]
            self.gbest_fit = max_fit
            self.gbest_total_risk = self.total_risk[max_fit_indx]

    def update_pbest(self):
        new_pbest = self.pbest_portfolio.copy()
        for i in range(self.num_of_portfolios):
            if self.fit[i] > self.pbest_fit[i]:
                self.pbest_fit[i] = self.fit[i]
                new_pbest[i] = self.portfolios[i]
        self.pbest_portfolio = new_pbest

    def update_expected_risk(self, c1=2, c2=2):
        # this function updates the expected risk according to some equations will be implemeted in the below
        # c1 & c2 are two acceleration constants
        new_expected_risk = np.zeros_like(self.expected_risk)
        r1 = np.random.random(size=(1, self.num_of_variables))
        r2 = np.random.random(size=(1, self.num_of_variables))
        # r1 & r2 : two arrays of random numbers between 0,1
        for i in range(self.num_of_portfolios):
            for j in range(self.num_of_variables):
                new_expected_risk[i][j] = self.expected_risk[i][j] + c1*r1[0][j]*(
                    self.pbest_portfolio[i][j]-self.portfolios[i][j]) + c2*r2[0][j]*(self.gbest_portfolio[j]-self.portfolios[i][j])

        self.expected_risk = new_expected_risk

    def update_portfolio(self):
        new_portfolios = np.zeros_like(self.portfolios)
        for i in range(self.num_of_portfolios):
            for j in range(self.num_of_variables):
                new_portfolios[i][j] = self.portfolios[i][j] + self.expected_risk[i][j]
        self.portfolios = new_portfolios

    def PSO(self, num_of_generation):
        # this is the main function which combines all functions to implement PSO algorithm
        # num_of_generation : the number of generations ,used as a stopping criteria
        portfolio = portfolio_wealth_optimization(self.num_of_portfolios, self.num_of_variables, self.x_min, self.x_max,
                                                  self.max_expected_return, self.min_total_expected_return, self.max_total_portfolio_risk, self.C, self.D)
        portfolio.initial_pop()
        # portfolio.Satisfy_all_constraints()
        portfolio.calculate_risk()  # velocity
        portfolio.pbest_portfolio = portfolio.portfolios.copy()
        portfolio.calculate_expected_return()  # fitness
        portfolio.pbest_fit = portfolio.fit.copy()
        portfolio.update_gbest()
        portfolio.Satisfy_all_constraints()

        for i in range(num_of_generation):
            portfolio.update_expected_risk()  # update velocity
            portfolio.calculate_risk()  # velocity
            portfolio.Satisfy_all_constraints()
            portfolio.update_portfolio()  # update portfolio
            portfolio.Satisfy_all_constraints()
            portfolio.calculate_expected_return()  # fitness
            portfolio.Satisfy_all_constraints()
            portfolio.update_pbest()  # update personal best portfolio
            portfolio.Satisfy_all_constraints()
            portfolio.update_gbest()  # update global best portfolio
            portfolio.Satisfy_all_constraints()  # handle constraints
        # Print the results
        print("the final best solution results are : \n")
        print('Global Best portfolio : ', portfolio.gbest_portfolio, "\n")
        print('return of the global best portfolio : ',
              max(portfolio.pbest_fit), "\n")
        # print('Average Best return Value: ',sum(portfolio.pbest_fit)/len(portfolio.pbest_fit), "\n")
        print("risk of the global best portfolio : ", portfolio.gbest_total_risk)


"""""
example 1 :

10 portfolios
5 decision variables
min limits for the weights : [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
max limits for the weights : [0.99, 0.99, 0.99, 0.99, 0.99]
max expected return for each asset : [50, 50, 50, 50, 50]
min total expected return : 5
max total expected risk : 5
C : [1, 2, 3, 4, 5]
D : [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
number of iterations : 200


"""
# example1 = portfolio_wealth_optimization(10, 5, [0.0001, 0.0001, 0.0001, 0.0001, 0.0001], [0.99, 0.99, 0.99, 0.99, 0.99], [50, 50, 50, 50, 50], 5, 5, [1, 2, 3, 4, 5], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
# print(example1.PSO(200))

"""""
example 2 :

100 portfolios
7 decision variables
min limits for the weights : [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
max limits for the weights : [0.99, 0.99, 0.99, 0.99, 0.99]
max expected return for each asset : [50, 50, 50, 50, 50, 50, 50]
min total expected return : 5
max total expected risk : 5
C : [2,4,6,8,1,2,1]
D : [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]
number of iterations : 200


"""
# example2 = portfolio_wealth_optimization(100, 7, [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001], [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99], [50, 50, 50, 50, 50, 50, 50], 5, 5, [2, 4, 6, 8, 1, 2, 1], [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]])
# print(example2.PSO(200))


"""""
example 3 :

50 portfolios
10 decision variables
min limits for the weights : [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
max limits for the weights : [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
max expected return for each asset : [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
min total expected return : 5
max total expected risk : 5
C : [2,4,6,8,1,2,1,1,2,2]
D : [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
number of iterations : 200


"""
example3 = portfolio_wealth_optimization(50, 10, [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001], [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99], [50, 50, 50, 50, 50, 50, 50, 50, 50, 50], 5, 5, [2, 4, 6, 8, 1, 2, 1, 1, 2, 2], [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [
                                         1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
print(example3.PSO(200))
