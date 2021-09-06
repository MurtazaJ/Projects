import argparse
import numpy as np
import pickle
# input_user()
parser = argparse.ArgumentParser()
# def arguments():
# parser = argparse.ArgumentParser(description='Test data of the new user to predict heart attack')
parser.add_argument('--statement', metavar='s', help = 'Test data of the new user to predict heart attack', type = str, required = False)
parser.add_argument('--age', help= 'Age of the Customer', type = float, metavar='a', required=False)
parser.add_argument('--Sex', help= '0 for male, 1 for female', type = float, metavar='s', required=False)
parser.add_argument('--cp', help= ' Chest Pain type chest pain type * Value 1: typical angina * Value 2: atypical angina * Value 3: non-anginal pain * Value 4: asymptomatic * trtbps', type = float, metavar='cp', required=False)
parser.add_argument('--trtbps', help= 'Enter a value', type = float, metavar='o',required=False)
parser.add_argument('--chol', help= 'Enter Cholestrol of the person', type = float, metavar='ch', required=False)
parser.add_argument('--fbs', help= 'Fbs Value 1 or 0', type = float, metavar='f', required=False)
parser.add_argument('--restecg', help= 'restecg Value 1 or 0', type = float, metavar='r', required=False)
parser.add_argument('--thalachh', help= 'thalachh enter a value', type = float, metavar='t', required=False )
parser.add_argument('--exng', help= 'exng Value 1 or 0',type = float,metavar='t', required=False) 
parser.add_argument('--oldpeak', help= 'oldpeak enter a value', type = float,metavar='t', required=False )
parser.add_argument('--slp', help= 'slp Value 1 or 0 or 2', type = float, metavar='t', required=False )
parser.add_argument('--caa', help= 'caa Value 1 or 0', type = float, metavar='t', required=False )
parser.add_argument('--thall', help= 'thall Value 1 or 0 or 3 or 4',type = float, metavar='t', required=False )
args = parser.parse_args()

if __name__ == '__main__':
    x1 = int(input('age of customer: '))
    x2 = int(input('Sex of customer: 0 for male, 1 for female: '))
    x3 = int(input('Chest Pain type chest pain type * 1: typical angina  2: atypical angina 3: non-anginal pain 4: asymptomatic:'))
    x4 = int(input('trtbps of customer: Enter a value: '))
    x5 = int(input('Enter Cholestrol of the person: '))
    x6 = int(input('Fbs Value 1 or 0: '))
    x7 = int(input('restecg Value 1 or 0: '))
    x8 = int(input('thalachh enter a value: '))
    x9 = int(input('exng Value 1 or 0: '))
    x10 = int(input('oldpeak enter a value: '))
    x11 = int(input('slp Value 1 or 0 or 2: '))
    x12 = int(input('caa Value 1 or 0: '))
    x13 = int(input('thall Value 1 or 0 or 3 or 4: '))

    args.age = x1
    args.Sex = x2
    args.cp = x3
    args.trtbps = x4
    args.chol = x5
    args.fbs = x6
    args.restecg = x7
    thalachh = x8
    args.exng = x9
    args.oldpeak = x10
    args.slp = x11
    args.caa = x12
    args.thall = x13
    

user_data = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,x11, x12, x13]
print(user_data)

with open('model_pickle', 'rb') as f:
    mp = pickle.load(f)
    mp_predict = mp.predict(np.array([user_data]))
    if mp_predict ==1:
        print('you can have a heartattack')
    else:
        print('you are safe')

