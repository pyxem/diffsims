#!/bin/bash
cd "$(dirname "$0")"
cd diffsims/tests
for folder in test_generators test_library test_sims test_utils
	do
	cd $folder
	autopep8 *.py --aggressive --in-place --max-line-length 130
	cd ..
done
cd ../
for folder in generators libraries sims utils
	do
	cd $folder
	autopep8 *.py --aggressive --in-place --max-line-length 130
	cd ..
done
