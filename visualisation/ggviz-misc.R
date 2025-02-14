library(tidyverse)


apiresponse <- read_csv("../data_raw/monthly_search_interest_data_tickers.csv")

selection <- apiresponse %>% 
  
  select(year_month= date, search_interest = value, search_term=keyword) %>% 
  distinct(year_month, search_term, .keep_all=T) %>% 
  mutate(year_month=lubridate::ceiling_date(year_month,"month")) 




ggplot(selection, mapping = aes(x=year_month, y=search_interest)) + geom_line(alpha=0.3)+
  facet_wrap(~search_term, ncol = 3)

ggsave("trends_final.png", width=12, height=100, units="cm")




# trying to match duolingo + etc... with the tickers -- but some are missing...

new_tickers <- tickers %>% select(ticker_symbol) %>% 
  mutate(tick2  = map(ticker_symbol, function(x) rep(x,108 ))) %>% unnest() %>% select(ticker_symbol)



# visualise prices 



ggplot(prices_daily, mapping = aes(x=date, y=adjusted)) + geom_line(alpha=0.3)+
  facet_wrap(~symbol, ncol = 3, scales = "free_y") 

ggsave("all_prices.png", width=12, height=100, units="cm")




