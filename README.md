# PSICalc Algorithm Package

This is a package for clustering Multiple Sequence Alignments (MSAs) utilizing normalized mutual information to examine protein subdomains. A complete data visualization tool for psicalc is available on the releases page. 

As an example:

```
import psicalc as pc

file = "<your_fasta_file>" # e.g "PF02517_seed.txt"

data = pc.read_txt_file_format(file) # read Fasta file

data = pc.durston_schema(data, 1) # Label column index starting at 1

result = pc.find_clusters(1, data) # will sample every column against msa

# Optionally write dictionary to csv
pc.write_output_data(1, result)
```

The program will run and return a csv file with the strongest clusters found in the MSA provided.
