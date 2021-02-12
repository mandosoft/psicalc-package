import glob
import psicalc as pc
a = 1
dir_name = 'Alignment CSV Files'

types = ['*.csv', '*.CSV']
f = []
for e in types:
    f.extend(glob.glob(f'{dir_name}/{e}'))

for i in range(a):
    df = pc.read_csv_file_format(f[i])
    print(df)
    pc.find_clusters(2, df)


