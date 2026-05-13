import pandas as pd
import os

def simple_convert_to_csv(file_path):

    base_name = os.path.splitext(file_path)[0]
    output_path = f"{base_name}.csv"


    df = pd.read_csv(file_path, skipinitialspace=True)
    df.to_csv(output_path, index=False)
    

simple_convert_to_csv('320_CLNF_AUs.txt')
simple_convert_to_csv('321_CLNF_AUs.txt')
simple_convert_to_csv('322_CLNF_AUs.txt')
simple_convert_to_csv('324_CLNF_AUs.txt')
simple_convert_to_csv('325_CLNF_AUs.txt')
simple_convert_to_csv('326_CLNF_AUs.txt')
simple_convert_to_csv('327_CLNF_AUs.txt')
simple_convert_to_csv('328_CLNF_AUs.txt')
simple_convert_to_csv('319_CLNF_AUs.txt')
# ...