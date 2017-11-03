# def main():
#     import pickle
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     accuracy = pickle.load(open('accuracy_with_kind_of_keep_prob.pkl', 'rb'))

#     for i in np.linspace(0.1, 1.0, 10):
#         plt.plot(np.linspace(1, 301, 31), accuracy[str(i)], label=str(i))

#     plt.legend(loc='best')
#     plt.xlabel('data')
#     plt.ylabel('accuracy')
#     plt.show()
    
# if __name__ == '__main__':
#     main()
