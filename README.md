# summarizer
Text summarisation and keyphrase extraction utility based on unsupervised learning. Three algorithms are implemented:

* Text summarization: TextRank (using word embeddings)
* Keyphrase extraction: TextRank
* Keyphrase extraction: SGRank

This work is based on two papers:

* [Mihalcea et al. 2004: TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
* [Danesh et al. 2015: SGRank: Combining Statistical and Graphical Methods to Improve the State of the Art in Unsupervised Keyphrase Extraction](http://www.aclweb.org/anthology/S15-1013)

## Usage


> python summarizer.py https://www.theguardian.com/environment/2016/nov/30/eu-declares-war-on-energy-waste-and-coal-subsidies-in-new-climate-package

> Europe will phase out coal subsidies and cut its energy use by 30% before the end of the next decade, under a major clean energy package announced in Brussels on Wednesday.  
>The 1,000-page blueprint to help the EU meet its Paris climate commitments also pencils in measures to cut electricity bills, boost renewable energies and limit use of unsustainable bioenergies.  
>The EU’s climate commissioner, Miguel Arias Cañete, said that the new energy efficiency target was a centrepiece of the package, and would curb energy imports, create jobs and bring down emissions.  
>Keyphrases: ['major clean energy package', 'new energy efficiency target', 'new green tech job', 'coal subsidy', 'new clean energy measure']

**Options**
* -s --sentences: Number of sentences in the summary
* -k --keyphrases: Number of keyphrases returned


## Dependencies

* spaCy (natural language processing)
* Networkx (graph representations and PageRank)
* requests (web scraping)
* BeautifulSoup 4 (html parsing)

