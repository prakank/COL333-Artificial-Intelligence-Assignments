## Automated Nurse Rostering Schedule

### Inputting Parameters

Python script </br>
Parameter file (**input_a.csv** or **input_b.csv**) </br>
Part number ('a' or 'b')</br>
Part Number is combined in the name of the input file </br>


### Execution

For execution
```sh
python3 A2.py InputFile_PartNum.csv
```

### Testing 

The solution obtained is dumped into **solution.json** file </br>
We have `tester.py` script which loops over a set of parameters and executes A2.py </br>
Output of `tester.py` can be stored in output file using:
```sh
python3 tester.py > output.txt
```

