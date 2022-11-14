# 558Project3


## Purpose  
This report contains a set of predictive models with automating mechanism. The data to be analyzed is the Online News Popularity Data Set summarizing a heterogeneous set of features about articles published by Mashable in a period of two years. The goal is to predict the number of shares in social networks (popularity). The dataset contains 39,644 observations with 61 variables.  
  
## R Packages Used  

The R packages used in this repo are listed below:
* `tidyverse`: the tidyverse is an opinionated collection of R packages designed for data science.  
* `caret`: a pretty powerful machine learning library in R.  
* `corrplot`: a package provides a visual exploratory tool on correlation matrix that supports automatic variable reordering to help detect hidden patterns among variables.  
* `gbm`: an implementation of extensions to Freund and Schapire's AdaBoost algorithm and Friedman's gradient boosting machine.  
* `randomForest`: a package used to create and analyze random forests.  
* `rmarkdown`: a package turn your analyses into high quality documents, reports, presentations and dashboards with R Markdown.  
  
## Links to Generated Analyses  

Here are the links to the generated analyses:  
* The analysis for [Lifestyle articles is available here](Lifestyle.html).  
* The analysis for [Entertainment articles is available here](Entertainment.html). 
* The analysis for [Business articles is available here](Business.html). 
* The analysis for [SocialMedia articles is available here](SocialMedia.html). 
* The analysis for [Tech articles is available here](Tech.html). 
* The analysis for [World articles is available here](World.html). 
  

## Code for Automation of Reports  

There are two steps involved in the creation of analyses from a single .Rmd file. The first step is to create file names to output to and a list with each channel name for using in `render()`. During this step we also need to use the `rm()` function to remove existing params from the R environment so that the document will knit properly for each iteration. The second step is to generate the reports for each channel.  

The code below generates file names and a list for each channel and stores this information in a tibble.  

```{r}
#Creating names for the channels for params
channelIDs<-c("Lifestyle", "Entertainment", "Business", "SocialMedia", "Tech", "World")

#Creating filenames
output_file<- paste0(channelIDs, ".md")

#Creating a list for each channel with just the channel parameter
parameters = lapply(channelIDs, FUN = function(x){list(channel = x)})

#Put into a data frame
reports<- tibble(output_file, parameters)

#Removing existing params from environment so that render will work properly
rm(params)
```


We can now knit using `apply()` to automatically generate reports for each data channel using the code below.  

```{r, eval=FALSE}
apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "Project3RMD.Rmd", output_file = x[[1]], params = x[[2]])
      })
```
