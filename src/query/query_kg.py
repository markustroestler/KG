from rdflib import Graph
from datetime import date, timedelta

# KG laden – falls Trends/News in separaten Files sind, beide parsen
g = Graph()
g.parse("data/kg_assets.ttl", format="turtle")
g.parse("data/kg_trends.ttl", format="turtle")
g.parse("data/kg_news.ttl", format="turtle")

cutoff = (date.today() - timedelta(days=7)).isoformat()  # z.B. '2025-08-04'

q2 = f"""
PREFIX ex:  <http://example.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?asset ?from ?to ?spark
       (COUNT(?n) AS ?posNews)
       (SUM(IF(?s >= 0.6, 1, 0)) AS ?strongPos)
WHERE {{
  # neuester Up-Trend pro Asset
  ?asset ex:hasTrend ?t .
  ?t ex:trendType ex:Up ; ex:fromDate ?from ; ex:toDate ?to ; ex:sparkline ?spark .
  FILTER NOT EXISTS {{
    ?asset ex:hasTrend ?t2 .
    ?t2 ex:trendType ex:Up ; ex:toDate ?to2 .
    FILTER(?to2 > ?to)
  }}

  # News-Link: klappt mit/ohne Backlink
  {{ ?asset ex:hasNews ?n }} UNION {{ ?n ex:aboutAsset ?asset }} .
  ?n ex:publishedDate ?d ;
     ex:sentiment ex:Positive ;
     ex:sentimentScore ?s .

  FILTER(?d >= "{cutoff}"^^xsd:date)
}}
GROUP BY ?asset ?from ?to ?spark
HAVING (COUNT(?n) >= 1)
ORDER BY DESC(?strongPos) DESC(?posNews)
"""

for asset, d_from, d_to, spark, posNews, strongPos in g.query(q2):
    print(
        asset.split("/")[-1],
        f"{d_from.toPython()}→{d_to.toPython()}",
        f"spark={str(spark)}",
        f"posNews={int(posNews)}",
        f"strongPos={int(strongPos)}"
    )
