es gibt aria2c Ich habe die links der website rausgeparsed und dann eine downloadconfig für arai2c erstellt
das lädt dann alles runter
das sind die files auf der website



# create aria2_downloads.txt
python app/epo/docdb.py -i app/epo/data_full_text.json  -o app/epo/aria2_downloads.txt -p 32

# aria download
aria2c -i app/epo/aria2_downloads.txt -s 16 -j 4 -c -d data/epo/full_text
`-s 16` → split each file into 16 segments for faster download
`-j 4` → download 4 files in parallel
`-c` → skip already present files

# load documents into elastic search
python app/epo/load_documents.py data/epo/full_text

# Elastic
## Init
sh app/epo/start-local.sh

## start/stop
app/epo/elastic-start-local/start.sh

## delete elastic index
curl -u username:password -X DELETE "http://localhost:9200/patents" 

## Search
python app/epo/search.py

## Show Pretty
curl -u username:password -X GET "localhost:9200/patents/_mapping?pretty"
curl -u username:password -X GET "localhost:9200/patents/_search?pretty"

# check size
du -h --max-depth=1 | sort -h

