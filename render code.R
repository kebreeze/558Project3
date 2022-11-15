# Automation  

#We first need to create file names to output to and a list with each channel name for using in `render()`. 


#Creating names for the channels for params
channelIDs<-c("Lifestyle", "Entertainment", "Business", "SocialMedia", "Tech", "World")
#Creating filenames
output_file<- paste0(channelIDs, ".md")
#Creating a list for each channel with just the channel parameter
parameters = lapply(channelIDs, FUN = function(x){list(channel = x)})
#Put into a data frame
reports<- tibble(output_file, parameters)



#We can now knit using `apply()` to automatically generate reports for each data channel.  

apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "Project3RMD.Rmd", 
               output_file = x[[1]], 
               params = x[[2]],
               output_options = list(
                 toc = TRUE,
                 toc_depth = 3
               ))
      })
