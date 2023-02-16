// Daniel Locke, 2017
// 
// Command line tool for creating documents from JSON files obtained from 
// www.courtlistener.com/api. Both Opinion and Cluster files are required. 
// These files are combined into an output json file as per the `Doc` 
// struct.  

package main

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"flag"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"strconv"
	"time"
)

type Opinion struct {
	Cluster string `json:"cluster"`
	ResourceUri string `json:"resource_uri"`
	AbsoluteUrl string `json:"absolute_url"`
	PlainText string `json:"plain_text"`
	Html string `json:"html_with_citations"`
	Cited []string `json:"opinions_cited"`

}

type Cluster struct {
	OpinionId int `json:""`
	DateFiled string `json:"date_filed"`
	CaseName string `json:"case_name"`
	CitationCount      int    `json:"citation_count"`
}

type ClusterCitations struct {
	FederalCiteOne     string `json:"federal_cite_one,omitempty"`
	FederalCiteThree   string `json:"federal_cite_three,omitempty"`
	FederalCiteTwo     string `json:"federal_cite_two,omitempty"`
	LexisCite          string `json:"lexis_cite,omitempty"`
	NeutralCite        string `json:"neutral_cite,omitempty"`
	PrecedentialStatus string `json:"precedential_status,omitempty"`
	ScotusEarlyCite    string `json:"scotus_early_cite,omitempty"`
	SpecialtyCiteOne   string `json:"specialty_cite_one,omitempty"`
	StateCiteOne       string `json:"state_cite_one,omitempty"`
	StateCiteRegional  string `json:"state_cite_regional,omitempty"`
	StateCiteThree     string `json:"state_cite_three,omitempty"`
	StateCiteTwo       string `json:"state_cite_two,omitempty"`
	WestlawCite        string `json:"westlaw_cite,omitempty"`
}

type Doc struct {
	Id int `json:"id"`
	Cluster int `json:"cluster"`
	Title string `json:"title"`
	Jurisdiction string `json:"jurisdiction"`
	DateFiled string `json:"date_filed"`
	CitationCount int `json:"citation_count"`
	Citations ClusterCitations `json:"citations"`
	Cited []int `json:"cited"`
	Html string `json:"html"`

}

type List struct {
	M map[string]int
}

func (l * List) listWrite(k string) {
	l.M[k] = 0
}

const LIST_FILE string = "complete.list"

var re = regexp.MustCompile("[^A-Za-z0-9[\\]().§]+")

func (l *List) Save() error {
	var buff bytes.Buffer
	enc := gob.NewEncoder(&buff)

	err := enc.Encode(l)
	if err != nil {
		return err
	}

	return ioutil.WriteFile(LIST_FILE, buff.Bytes(), 0664)
}

func load(file string) (*List, error) {
	var data = make([]byte, 0)
	var l List

	data, err := ioutil.ReadFile(file)
	if os.IsNotExist(err) {
		_, err = os.Create(file)
		l.M = make(map[string]int)
		return &l, err
	} else {
		if err != nil {
			return nil, err
		}
		dec := gob.NewDecoder(bytes.NewReader(data))

		err = dec.Decode(&l)
		if err != nil {
			return nil, err
		}
	}

	return &l, nil
}

func OpenJsonFile(filepath string) ([]byte, error) {
	// return os.OpenFile(filepath, os.O_RDWR, 0755)
	data, err := ioutil.ReadFile(filepath)
	if err != nil {
		return nil, err
	}
	return data, nil
}

func getDir(dir string) string {
	path, _ := filepath.Split(dir)

	if path[len(path)-1] == '/' {
		path = strings.TrimRight(path, "/")
	}
	p := strings.Split(path, "/")
	return p[len(p)-1]
}

func generateDocs(source, dest, clusterPath string, list *List) error {
	sourceInfo, err := os.Stat(source)
	if err != nil {
		return err
	}

	err = os.MkdirAll(dest, sourceInfo.Mode())
	if err != nil {
		return err
	}

	dir, err := os.Open(source)
	if err != nil {
		return err
	}

	obj, err := dir.Readdir(-1)
	if err != nil {
		return err
	}

	for i := range obj {

		sourcePath := source + "/" + obj[i].Name()
		destPath := dest + "/" + obj[i].Name()
		if obj[i].Name() != ".DS_Store" {
				if obj[i].IsDir() {
				clustPath := clusterPath + "/" + obj[i].Name()
				err = generateDocs(sourcePath, destPath, clustPath, list)
				if err != nil {
					return err
				}
			} else {
				if _, ok := list.M[obj[i].Name()]; !ok {
					err = GenerateDoc(sourcePath, obj[i].Name(), destPath, clusterPath)
					if err != nil {
						return err
					}
					list.listWrite(obj[i].Name())
				}
			}
		}

	}
	return list.Save()
}

func GenerateDoc(source, sourceFile, dest, clusterPath string) error {
	log.Println(sourceFile)
	doc := Doc{}
	op_file, err := OpenJsonFile(source)
	if err != nil {
		log.Panic(err)
		return err
	}
	op := Opinion{}
	err = json.Unmarshal(op_file, &op)
	if err != nil {
		log.Panic(err)
		return err
	}

	doc.Id, err = strconv.Atoi(strings.Replace(sourceFile, ".json","", -1))
	if err != nil {
		log.Panic(err)
		return err
	}

	doc.Jurisdiction = getDir(source)
	doc.Html = op.Html


	for _, citation := range op.Cited {
		cit := strings.Replace(citation, "http://www.courtlistener.com/api/rest/v3/opinions/", "", 1)
		cit = strings.Replace(cit, "/", "", 1)
		c, err := strconv.Atoi(cit)
		if err != nil {
			log.Panic(err)
			return err
		}
		doc.Cited = append(doc.Cited, c)
	}

	cluster := strings.Replace(op.Cluster, "http://www.courtlistener.com/api/rest/v3/clusters/", "", 1)
	cluster = strings.Replace(cluster, "/", "", 1)
	doc.Cluster, err = strconv.Atoi(cluster)
	if err != nil {
		log.Panic(err)
		return err
	}

	clust_file_name := cluster + ".json"

	clust_file, err := OpenJsonFile(clusterPath + "/" + clust_file_name)
	if os.IsNotExist(err) {
		return nil
	} else if err != nil {
		log.Panic(err)
		return err
	}

	c := Cluster{}
	err = json.Unmarshal(clust_file, &c)
	if err != nil {
		log.Panic(err)
		return err
	}

	clustCitations := ClusterCitations{}
	err = json.Unmarshal(clust_file, &clustCitations)
	if err != nil {
		log.Panic(err)
		return err
	}

	doc.Title = c.CaseName
	doc.DateFiled = c.DateFiled
	doc.CitationCount = c.CitationCount
	doc.Citations = clustCitations

	tmp, err := json.MarshalIndent(doc, "", "  ")
	// tmp, err := json.Marshal(doc)
	if err != nil {
		log.Panic(err)
		return err
	}

	doc_file, err := os.OpenFile(dest, os.O_CREATE|os.O_RDWR, 0755)
	if err != nil {
		log.Panic(err)
		return err
	}
	defer doc_file.Close()

	doc_file.Write(tmp)

	return nil
}

func main() {
	start := time.Now()
	fmt.Println(`

====================================
Collection Doc Creation

	`)

	opPath := flag.String("op", "", "Path to opinions")
	cPath := flag.String("c", "", "Path to clusters")
	oPath := flag.String("o", "", "Output path")
	flag.Parse()

	list, err := load(LIST_FILE)
	if err != nil {
		log.Panic(err)
	}

	err = generateDocs(*opPath, *oPath, *cPath, list)
	if err != nil {
		defer log.Panic(err)
		err = list.Save()
		if err != nil {
			log.Panic(err)
		}
	}
	
	if err != nil {
		log.Panic(err)
	}
	fmt.Println("Completed in", time.Since(start))
}
