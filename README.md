# 558Project3

1.  A brief description of the purpose of the repo  
2.  A list of R packages used  
3.  Links to the generated analyses  
4.  The code used to create the analyses from a single .Rmd file (i.e. the render() code)  


This report contains a set of predictive models with automating mechanism. The data to be analyzed is the Online News Popularity Data Set summarizing a heterogeneous set of features about articles published by Mashable in a period of two years. The goal is to predict the number of shares in social networks (popularity). The dataset contains 39,644 observations with 61 variables.  
  

The R packages used in this repo are listed below:
* `tidyverse`: the tidyverse is an opinionated collection of R packages designed for data science.  
* `caret`: a pretty powerful machine learning library in R.  
* `corrplot`: a package provides a visual exploratory tool on correlation matrix that supports automatic variable reordering to help detect hidden patterns among variables.  
* `gbm`: an implementation of extensions to Freund and Schapire's AdaBoost algorithm and Friedman's gradient boosting machine.  
* `randomForest`: a package used to create and analyze random forests.  
* `rmarkdown`: a package turn your analyses into high quality documents, reports, presentations and dashboards with R Markdown.  
  

Here are the links to the generated analyses:  
* The analysis for [Lifestyle articles is available here](Lifestyle.html).  
* The analysis for [Entertainment articles is available here](Entertainment.html). 
* The analysis for [Business articles is available here](Business.html). 
* The analysis for [SocialMedia articles is available here](SocialMedia.html). 
* The analysis for [Tech articles is available here](Tech.html). 
* The analysis for [World articles is available here](World.html). 
  

The code used to create the analyses from a single .Rmd file is shown below:

```{r}
library(rmarkdown)
rmarkdown::render(input = "Lifestyle.md", output_file = "Lifestyle.html")
rmarkdown::render(input = "Entertainment.md", output_file = "Entertainment.html")
rmarkdown::render(input = "Business.md", output_file = "Business.html")
rmarkdown::render(input = "SocialMedia.md", output_file = "SocialMedia.html")
rmarkdown::render(input = "Tech.md", output_file = "Tech.html")
rmarkdown::render(input = "World.md", output_file = "World.html")
```
