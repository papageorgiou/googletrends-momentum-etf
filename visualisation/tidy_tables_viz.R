library(tidyverse)
library(ggrepel)
library(scales)
library(ggthemes)
library(patchwork)
library(kw)
library(extrafont)
loadfonts(device = "win")

# library(showtext)
# font_add_google("Ubuntu")
# showtext_auto()

etf_name <- "GOOGLE TRENDS ETF \n (inspired by G. Allsopp's SEO ETF)"

etf <- read_csv("../data_proc/df_index.csv") %>% select(date, investment_value) %>% mutate(index=etf_name)
nasd <- read_csv("../data_proc/comparison_data_nasdaq.csv")%>% select(date, investment_value) %>% mutate(index="NASDAQ ETF")
snp <- read_csv("../data_proc/comparison_data_snp.csv")%>% select(date, investment_value) %>% mutate(index="S&P 500 ETF")

#????

all_index = bind_rows(etf, nasd, snp)  #%>% mutate(investment_value=investment_value/10)

caption_text <- "Sources: Curated list of digital-first public startups from gaps.com/public. Yfinance & Google Trends data Jan 2015- Dec 2024."

theme_set(new =ggthemes::theme_fivethirtyeight() ) 
#ggthemes::theme_fivethirtyeight((base_family = "Roboto Condensed"))
#kw::my_social_theme()
# theme_minimal(base_family = "Roboto Condensed")
# #theme( text = element_text(family = "Ubuntu	"))
# ggthemes::theme_economist_white()

plota <- ggplot(data = all_index, aes(x = date, y=investment_value, color=index)) +  
 geom_line(linewidth=0.8) +
  labs(x = NULL,
       y = NULL,
       title = "Evolution of $100,000 invested in 2015 by Fund",
       #subtitle = "\U0001F50D Google Trends Momentum ETF 10x's the initial investment", 
       caption =caption_text) + 
  scale_x_date(limits = c(ymd("2015-01-01"),ymd("2025-01-01")),expand = c(0,0)) +
  scale_y_continuous(limits = c(0, 1100000), expand = c(0, 0), breaks = c(0, 100000,200000, 400000, 600000, 800000, 1000000), 
                     labels = scales::label_dollar(scale = 1, trim = TRUE))  +
  scale_color_manual(name=NULL, breaks = c("S&P 500 ETF", "NASDAQ ETF", "GOOGLE TRENDS ETF \n (inspired by G. Allsopp's SEO ETF)" ), 
                     values = c("GOOGLE TRENDS ETF \n (inspired by G. Allsopp's SEO ETF)"= "#EA4335", 
                                "NASDAQ ETF"= "grey50", "S&P 500 ETF"= "gray60")) + 
                       theme(legend.position="top", 
                             plot.caption = element_text(size = 8), 
                             plot.title = element_text(size = 15, face = "bold"),
                             plot.subtitle = element_text(size = 10),
                             #panel.grid.major.y = element_line(linetype = "dotted", colour = "gray50", size = 0.3), 
                             panel.grid.major.x = element_blank()) +  #ggthemes::theme_economist_white()
  gghighlight::gghighlight(max_highlight = 3,   max(investment_value), use_direct_label = TRUE)

plota

ggsave("gtrends_etf.png", width = 8, height = 5)
ggsave("gtrends_etf_w8h6.png", width = 8, height = 6)
ggsave("gtrends_etf_w6.5h4.png", width = 6.5, height = 4)


# scale = 1e-3 converts values from 100,000 ??? 100 (to show in K format).
# suffix = "K" ensures numbers appear as "$100K" instead of "$100,000".
# gghighlight::gghighlight(max_highlight = 3, 
#                          max(investment_value), use_direct_label = TRUE) + 



stocks <- read_csv("../data_raw/prices_daily.csv")  
selected_stocks <- stocks %>% filter(ticker %in% c("BKNG", "INTU", "HUBS", "CVNA", "CPRT", "WDAY" ))

plotb <- ggplot(selected_stocks, aes(x = date, y=adjusted, colour=ticker)) + geom_line() +  
  facet_wrap(~ticker, nrow=3, scales = "free_y") + labs(title ="Top contributing stocks (adjusted prices)") + theme(legend.position="none")


final <- plota / plotb + plot_layout(nrow = 2, heights=c(2,1))



semesters <- read_csv("semester_stock_weights.csv") 
semesters %>% filter(`Weight (%)` > 0) %>%  count(Stock, sort=T) 
