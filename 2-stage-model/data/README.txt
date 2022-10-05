===================================================
README for the Dagstuhl-15512 ArgQuality Corpus
===================================================

This folder contains all files of the Dagstuhl-15512 ArgQuality Corpus. In case you publish any results related to the corpus, please cite the following paper:

	@inproceedings{wachsmuth:2017a,
		author    = {Henning Wachsmuth and Nona Naderi and Yufang Hou and Yonatan Bilu and Vinodkumar Prabhakaran and Tim Alberdingk Thijm and Graeme Hirst and Benno Stein},
		title     = {{Computational Argumentation Quality Assessment in Natural Language}},
		booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics},
		year      = {2017},
		location  = {Valencia, Spain},
	}

In case you have any questions regarding the corpus, don't hesitate to contact the authors of this paper. 

You will find the following data in the respective files and folders here:



---------------------------------------------------
./dagstuhl-15512-argquality-corpus-annotated-xmi/
---------------------------------------------------

This folder contains the complete annotated corpus in the Apache UIMA format. Each given XMI file stores one argument together with all its annotations. The XMI files are organized in subfolders according two the issues and stances the arguments refer to.

Besides the quality annotations, the XMI files also include all metadata given for the arguments in their sources. This metadata is described for the CSV format below.

General information about the XMI format can be found at http://uima.apache.org. The type system that defines the used annotation scheme is found in the folder ./uima-type-systems/ described below.




---------------------------------------------------
./dagstuhl-15512-argquality-corpus-annotated.csv
---------------------------------------------------

This tab-separated CSV file represents the complete annotated corpus. In particular, it contains all annotations of each of the three annotators for all 320 arguments.

The header line of the CSV file specifies the meanings of the respective columns:

	annotator	-> ID of the annotator ("1", "2", or "3")
	argumentative	-> Argumentative or not ("y" or "n")
	overall quality		-> Assigned score for overall quality ("1 (Low)", "2 (Average"), or "3 (High)")
	local acceptability	-> Assigned score for local accepability ("1 (Low)", "2 (Average"), or "3 (High)")
	...	-> Respective columns for all other dimensions from the taxonomy 	
	argument	-> The text of the argument (as specified in the UKPConvArgRank dataset)
	#id	-> The ID of the argument (as specified in the UKPConvArgRank dataset)
	issue	-> The issue the argument refers to (as specified in the respective file name of the UKPConvArgRank dataset)
	stance	-> The stance the argument refers to (as specified in the respective file name of the UKPConvArgRank dataset)	
Each content line of the CSV file contains all annotations of one argument. The lines are ordered according to the following columns:

	stance > issue > #id > annotator

Notice that, in case an annotator classified an argument as non-argumentative ("n"), no score is given for any dimension.  




---------------------------------------------------
./dagstuhl-15512-argquality-corpus-guidelines.pdf
---------------------------------------------------

This file contains the final annotation guidelines that the annotators used to annotate all 320 arguments in the corpus.




---------------------------------------------------
./dagstuhl-15512-argquality-corpus-source/
---------------------------------------------------

This folder contains the 32 original tab-separated CSV files from the UKPConvArgRank dataset, one for each combination of issue and stance (named accordingly).

Each CSV file contains the ID, rank, and text of all arguments that refer to the respective issue/stance combination. The CSV files contain between 25 and 35 arguments.

From each of these files, we selected 10 arguments as described in the paper. In one single file (william-farquhar-ought-to-be-honoured-as-the-rightful-founder-of-singapore_no-it-is-raffles.csv), the fourth and fifth highest ranked argument are duplicates. In this case, we selected the sixth highest ranked argument instead of the fifth. 




---------------------------------------------------
./dagstuhl-15512-argquality-corpus-typesystems/
---------------------------------------------------

This folder contains the Apache UIMA type systems that are needed to process the above-mentioned XMI files. 

In particular, the type system to be used is stored in the file "ArgumentationTypeSystem.xml". Notice, though, that this type system imports some of the other given type systems, so keep them all together.




---------------------------------------------------
./dagstuhl-15512-argquality-corpus-unannotated/
---------------------------------------------------

This folder contains 32 tab-separated CSV files with the 320 arguments that we selected for annotation, one file for each combination of issue and stance (named accordingly).

Each CSV file contains the ID, rank, and text of all arguments that refer to the respective issue/stance combination. Each file contains 10 arguments, ordered by ID.






