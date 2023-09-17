# Performs OCR on a list of images using Tesseract and performs fuzzy string matching with a given list of strings.

## Tested against Windows 10 / Python 3.11 / Anaconda

### pip install tesseractrapidfuzz


```python


This function takes a path to the Tesseract OCR executable, a list of image paths or URLs,
a list of strings to compare against the recognized text, and optional fuzzy matching settings.
It returns a pandas DataFrame with OCR results and fuzzy matching scores.

Args:
	tesseract_path (str): Path to the Tesseract OCR executable.
	allpics (Union[list, tuple]): List of image paths, URLs, or other image data sources.
	strings_to_compare (Union[list, tuple, np.ndarray]): List of strings for fuzzy matching.
	compare_single_words (bool, optional): Enable fuzzy matching on individual words.
		Defaults to True.
	compare_grouped_words (bool, optional): Enable fuzzy matching on grouped words.
		Defaults to True.
	scorer_single_words (valid_scorer, optional): Fuzzy matching scorer for single words.
		Defaults to "WRatio".
	scorer_grouped_words (valid_scorer, optional): Fuzzy matching scorer for grouped words.
		Defaults to "WRatio".
	add_after_tesseract_path (str, optional): Additional arguments for Tesseract after
		the input image path. Defaults to an empty string.
	add_at_the_end (str, optional): Additional arguments to append to the Tesseract command.
		Defaults to "-l eng --psm 3".
	**kwargs: Additional keyword arguments to control the fuzzy matching process.

Returns:
	pd.DataFrame: A DataFrame with OCR results and fuzzy matching scores, including columns:
		- 'id_img': Image ID
		- 'id_word': Word ID within the image
		- 'ocr_result': Recognized text
		- 'start_x': Starting X-coordinate of the bounding box
		- 'end_x': Ending X-coordinate of the bounding box
		- 'start_y': Starting Y-coordinate of the bounding box
		- 'end_y': Ending Y-coordinate of the bounding box
		- 'conf': Confidence score
		- 'grouped_text': Grouped text for fuzzy matching
		- 'compared_grouped_words_similarity': Fuzzy matching score for grouped words
		- 'compared_grouped_words_index': Index of the matched string for grouped words
		- 'compared_grouped_words_value': Matched value for grouped words
		- 'compared_single_words_similarity': Fuzzy matching score for single words
		- 'compared_single_words_index': Index of the matched string for single words
		- 'compared_single_words_value': Matched value for single words

Example:
	import re
	from tesseractrapidfuzz import ocr_and_fuzzy_check
	df = ocr_and_fuzzy_check(
		tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
		allpics=[
			"https://m.media-amazon.com/images/I/711y6oE2JrL._SL1500_.jpg",
			"https://m.media-amazon.com/images/I/61g+KBpG20L._SL1500_.jpg",
		],
		strings_to_compare=[
			"nonviolent",
			"communication",
			"emotional",
			"well-being",
			"terrible",
			"today.",
			"discover",
			"definitive",
			"guides",
			"transforming",
			"converting",
			"conflict",
			"meaningful",
			"connection,",
			"unveiling",
			"inspirational",
			"strategies",
			"engagement.",
			"martha",
			"williams",
			"nonviolent communication",
			"emotional well-being",
			"I had a terrible day at work today.",
			"wait till you",
			"heared about",
			"the art of nonviolent communication",
			"martha a. williams",
		],
		compare_single_words=True,
		compare_grouped_words=True,
		scorer_single_words="QRatio",
		scorer_grouped_words="WRatio",
		add_after_tesseract_path="",
		add_at_the_end="-l eng --psm 3",
		workers=5,
		processor=lambda x: re.sub(r"\W+", "", str(x).lower()),
	)
	print(df.to_string())
	# ...
	# 7        1        8       terrible      448    563      371    396    77       2875       505       383    115      25           2                                    | had a terrible                100.000000                    4            terrible                       90.0                     4                             terrible
	# 8        1        9            day      363    418      415    448    96       1815       390       431     55      33           3                                         day at work                 75.000000                    5              today.                       90.0                    22  I had a terrible day at work today.
	# 9        1       10             at      427    457      418    440    96        660       442       429     30      22           3                                         day at work                 50.000000                   18              martha                       90.0                    22  I had a terrible day at work today.
	# 10       1       11           work      466    540      415    440    96       1850       503       427     74      25           3                                         day at work                 33.333332                    6            discover                       90.0                    22  I had a terrible day at work today.
	# 11       1       12         today.      402    498      460    492    96       3072       450       476     96      32           4                                              today.                100.000000                    5              today.                      100.0                     5                               today.
	# 12       1       13           Wait      551    635      525    556    95       2604       593       540     84      31           5                                       Wait till you                 53.333332                   23       wait till you                      100.0                    23                        wait till you
	# 13       1       14           till      645    695      525    556    96       1550       670       540     50      31           5                                       Wait till you                 53.333332                   23       wait till you                      100.0                    23                        wait till you
	# 14       1       15            you      705    773      533    565    96       2176       739       549     68      32           5                                       Wait till you                 42.857143                   23       wait till you                      100.0                    23                        wait till you
	# 15       1       16           hear      562    645      579    610    95       2573       603       594     83      31           6                                          hear about                 53.333332                   24        heared about                       90.0                    24                         heared about
	# 16       1       17          about      663    767      579    610    96       3224       715       594    104      31           6                                          hear about                 62.500000                   24        heared about                       90.0                    24                         heared about
	# 17       2        1            ART       94    246      125    207    95      12464       170       166    152      82           7                                   ART OF NONVIOLENT                 66.666664                   18              martha                       90.0                     0                           nonviolent
	# 18       2        2             OF      275    376      125    207    95       8282       325       166    101      82           7                                   ART OF NONVIOLENT                 40.000000                   11            conflict                       90.0                     0                           nonviolent
	# 19       2        3     NONVIOLENT      407    907      125    206    96      40500       657       165    500      81           7                                   ART OF NONVIOLENT                100.000000                    0          nonviolent                       90.0                     0                           nonviolent
	# 20       2        4  COMMUNICATION      167    832      296    377    96      53865       499       336    665      81           8                                       COMMUNICATION                100.000000                    1       communication                      100.0                     1                        communication
	# 21       2        5            TAR      319    379      428    444    31        960       349       436     60      16           9                                                 TAR                 50.000000                    5              today.                       72.0                     9                         transforming
	# 22       2        6       DISCOVER      192    307      624    667    96       4945       249       645    115      43          10        DISCOVER THE DEFINITIVE GUIDES TO NONVIOLENT                100.000000                    6            discover                       90.0                     0                           nonviolent
	# 23       2        7            THE      320    360      624    667    96       1720       340       645     40      43          10        DISCOVER THE DEFINITIVE GUIDES TO NONVIOLENT                 44.444443                   18              martha                       90.0                     0                           nonviolent
	# 24       2        8     DEFINITIVE      374    507      624    667    96       5719       440       645    133      43          10        DISCOVER THE DEFINITIVE GUIDES TO NONVIOLENT                100.000000                    7          definitive                       90.0                     0                           nonviolent
	# 25       2        9         GUIDES      521    604      624    667    96       3569       562       645     83      43          10        DISCOVER THE DEFINITIVE GUIDES TO NONVIOLENT                100.000000                    8              guides                       90.0                     0                           nonviolent
	# 26       2       10             TO      618    645      628    654    96        702       631       641     27      26          10        DISCOVER THE DEFINITIVE GUIDES TO NONVIOLENT                 57.142857                    5              today.                       90.0                     0                           nonviolent
	# 27       2       11     NONVIOLENT      661    810      624    667    96       6407       735       645    149      43          10        DISCOVER THE DEFINITIVE GUIDES TO NONVIOLENT                100.000000                    0          nonviolent                       90.0                     0                           nonviolent
	# ...

Note:
	- The function combines OCR results with fuzzy string matching, allowing for versatile text analysis.
	- Valid_scoring options are: "WRatio", "QRatio", "ratio", "partial_ratio".
```