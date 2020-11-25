import matplotlib.pyplot  as plt


# # plot Part A q4
# q4_names = ['q1','q2', 'q3a', 'q3b', 'q3c', 'q3d']
# q4_values = [49.3,51.2,50.6,44.6,47.2,47.7]
# plt.title('test accuracies')
# plt.scatter(q4_names,q4_values)
# plt.savefig('a_q4_acc.png')
# plt.close()

# plot Part B q5
# normal_acc_names = ['q1','q2', 'q3','q4']
# normal_acc_values = [67.3,	50.2,	57.7,	73.97]
# plt.title('test accuracies')
# plt.scatter(normal_acc_names,normal_acc_values)
# plt.savefig('b_q5_normal_acc.png')
# plt.close()

# normal_time_names = ['q1','q2', 'q3','q4']
# normal_time_values = [2200.1104850769043,	277.6241948604584,	1462.3785090446472,	1826.6572318077087]
# plt.title('running time')
# plt.scatter(normal_time_names,normal_time_values)
# plt.savefig('b_q5_normal_time.png')
# plt.close()

# # Part B q5, no drop out
# no_acc_names = ['q1','q2', 'q3','q4']
# no_acc_values = [67.7,	31.0,	66.3,	87.0]
# plt.title('test accuracies')
# plt.scatter(no_acc_names,no_acc_values)
# plt.savefig('b_q5_no_dr_acc.png')
# plt.close()


# no_time_names = ['q1','q2', 'q3','q4']
# no_time_values = [2191,	278,	1385,	1705]
# plt.title('running time')
# plt.scatter(no_time_names,no_time_values)
# plt.savefig('b_q5_no_dr_time.png')
# plt.close()

# plot Part B q5
names = ['Vanilla', 'GRU', 'LSTM', '2 layers', 'Gradient clipping']
q3_value = [0.82, 57.7, 33.7,0.66,57.7]
q4_value = [0.71,73.9,55.6,0.71,73.9]
plt.title('test accuracies') 
q3 = plt.scatter(names,q3_value, marker='o')
q4 = plt.scatter(names,q4_value, marker = '*')
plt.legend([q3,q4],['q3','q4'], loc='center left')
plt.savefig('b_q6_acc.png')
plt.close()

# # plot Part B q5
t_names = ['Vanilla', 'GRU', 'LSTM', '2 layers', 'Gradient clipping']
q3_t_value = [647,1462,1159,2364,1404]
q4_t_value = [837,1826,1377,2730,1831]
plt.title('running time')
q3 = plt.scatter(names,q3_t_value, marker='o')
q4 = plt.scatter(names,q4_t_value, marker = '*')
plt.legend([q3,q4],['q3','q4'], loc='center left')
plt.savefig('b_q6_time.png')
plt.legend(loc = 'best')
plt.close()