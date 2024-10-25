clear all 
cd "/Users/owen/Covid5/"

import fred LNS14024230 CSUSHPINSA SP500, aggregate(monthly,eop)


rename LNS14024230 unem55
rename CSUSHPINSA housing 
rename SP500 sp500

gen year = year(daten)
gen mo = ym(year(daten),month(daten))
tsset mo 

* unem rate graph 
tsline unem55 if inrange(year,2000,2024), ///
	xtitle("") ///
	xla(`=tm(2000m1)'(120)`=tm(2020m1)',format(%tmCY)) /// 
	xsc(r(`=tm(2000m1)' `=tm(2024m9)')) /// 
	xline(`=tm(2020m3)', lc(black) lp(dot)) xsize(4) ///
	ytitle("")
graph export output/figs/ur55.pdf, replace 

foreach var in housing sp500 { 
	sum `var' if year(daten)==2020 & month(daten)==1
	replace `var' = `var'/r(mean)
}

* assets graph 
tsline housing sp500 if inrange(year,2015,2024), ///
	xtitle("") ///
	xla(`=tm(2016m1)'(24)`=tm(2024m1)',format(%tmCY)) /// 
	xsc(r(`=tm(2015m1)' `=tm(2024m1)')) /// 
	legend(order(1 "Case-Shiller housing index, indexed to Jan 2020" 2 "S&P 500, indexed to Jan 2020") pos(6)) ///
	xline(`=tm(2020m3)', lc(black) lp(dot)) xsize(4)
graph export output/figs/assets.pdf, replace 
