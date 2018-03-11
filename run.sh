if [[ $# -ne 4 ]]
then
echo "Enter Correct number of arguments"
exit 1
fi
QNO=$1
MNO=$2
IFN=$3
OFN=$4
if [ -e OFN ]
then
	rm $4
fi
if [ $QNO -eq 1 ]
then
	python naive_output.py $MNO $IFN $OFN
fi
if [ $QNO -eq 2 ]
then
	if [ $MNO -eq 1 ]
	then
		python svm_output.py $IFN $OFN
	fi
	if [ $MNO -eq 2 ]
	then
		python svm_output2.py $IFN
		svm-scale -l 0 -u 1 changed_format.txt > test_out_scale
		svm-predict test_out_scale ./pickles/svm2.model $OFN
	fi
	if [ $MNO -eq 3 ]
	then
		python svm_output2.py $IFN
		svm-scale -l 0 -u 1 changed_format.txt > test_out_scale
		svm-predict test_out_scale ./pickles/svm3.model $OFN
	fi
fi
