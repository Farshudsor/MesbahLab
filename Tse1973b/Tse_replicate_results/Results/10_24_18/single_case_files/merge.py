import os

csv_header = 'Timestamp,Client IP,Web Service,Status,Good,Bad'
csv_out = 'R10.csv'

csv_dir = os.getcwd()

dir_tree = os.walk(csv_dir)
for dirpath, dirnames, filenames in dir_tree:
    pass

csv_list = ['Results_10_C1.csv','Results_10_2C.csv','Results_10_c3.csv']
for file in filenames:
    if file.endswith('.csv'):
        csv_list.append(file)

csv_merge = open(csv_out, 'w')
csv_merge.write(csv_header)
csv_merge.write('\n')

for file in csv_list:
    csv_in = open(file)
    for line in csv_in:
        if line.startswith(csv_header):
            continue
        csv_merge.write(line)
    csv_in.close()
    csv_merge.close()
print('Verify consolidated CSV file : ' + csv_out)
