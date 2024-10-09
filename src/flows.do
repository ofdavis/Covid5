frame change default
cd "/users/owen/Covid5"
use data/covid_long.dta, clear
rename ur urate

qui sum mo 
global mo_max = r(max)
global covid_line "xline(`=tm(2020m4)', lp(dash) lc(black))"
global covid_line2 "xline(`=tm(2021m3)', lp(dash) lc(black))"

***** reconstruct monthly rates --50 plus only
frame copy default flows_rates_yr_50, replace
frame change flows_rates_yr_50

*create indicators for flows 
cap drop er ur nr re ru rn rr l_e l_u l_n l_r
gen er = L12.emp==1 & emp==4
gen ur = L12.emp==2 & emp==4
gen nr = L12.emp==3 & emp==4
gen re = L12.emp==4 & emp==1
gen ru = L12.emp==4 & emp==2
gen rn = L12.emp==4 & emp==3
gen rr = L12.emp==4 & emp==4

* create indicators for lagged labor force totals 
gen l_e = l12.emp==1
gen l_u = l12.emp==2
gen l_n = l12.emp==3
gen l_r = l12.emp==4
collapse (sum) er ur nr rr re ru rn l_e l_u l_n l_r if age>=55 [fw=wtfinl], by(mo) //[fw=yearwt] <-- need to redo 

drop if mo<=`=tm(2010m12)'

* create rates of transition to and from retirement 
foreach x in e u n {
	gen `x'r_rt = `x'r/l_`x'
	gen r`x'_rt = r`x'/l_r
}

* create net vars and net var rt, using prev retired pop as net rate
foreach var in e u n {
	gen net_`var'r = `var'r - r`var'
	gen net_`var'r_rt = net_`var'r/l_r
}

* alternative rate var -- use 50+ pop as denominator >>this is the one used in the graph
gen l_pop = l_e + l_u + l_n + l_r
foreach x in e u n {
	gen `x'r_rt2 = `x'r/l_pop
	gen r`x'_rt2 = r`x'/l_pop
}

* combine lf components 
gen r_lf_rt2 = re_rt2 + ru_rt2
gen lf_r_rt2 = er_rt2 + ur_rt2

* fig  with all ret flows 
frame change flows_rates_yr_50
tsline er_rt2 re_rt2 ur_rt2 ru_rt2 nr_rt2 rn_rt2, ///
	lc(navy navy maroon maroon forest_green forest_green) lp(solid shortdash solid shortdash solid shortdash) ///
	name(`frm'_2, replace) $covid_line $covid_line2 xtitle("") legend(off) ///
	text(0.034 755 "Emp-Ret", color(navy)) text(0.029  755 "NILF-Ret", color(forest_green))  text(0.019  755 "Ret-NILF", color(forest_green)) ///
	text(0.014 755 "Ret-Emp", color(navy)) text(0.0045 756 "Unem-Ret", color(maroon))  		 text(0.0015 756 "Ret-Unem", color(maroon)) ///
	xscale(r(`=tm(2011m1)' `=tm(2023m3)')) xlabel(`=tm(2011m1)'(12)$mo_max, format(%tmCY))

* fig without NILF-related flows  
frame change flows_rates_yr_50
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
tsline er_rt2 re_rt2 ur_rt2 ru_rt2 , ///
	lc("`r(p1)'" "`r(p1)'" "`r(p2)'" "`r(p2)'" ) lp(solid shortdash solid shortdash) ///
	name(`frm'_2, replace) $covid_line $covid_line2 xtitle("") legend(off) ///
	text(0.039 755 "Emp-Ret", color("`r(p1)'")) text(0.007 756 "Ret-Unem", color("`r(p2)'"))  ///
	text(0.018 755 "Ret-Emp", color("`r(p1)'")) text(0.00 756 "Unem-Ret", color("`r(p2)'"))  ///
	xscale(r(`=tm(2011m1)' `=tm(2023m3)')) xlabel(`=tm(2011m1)'(12)$mo_max, format(%tmCY))

* combined LF-ret 
quietly sum r_lf_rt2 if inrange(mo, `=tm(2017m1)', `=tm(2019m12)')
local r_lf_avg = r(mean)
di `r_lf_avg'
quietly sum lf_r_rt2 if inrange(mo, `=tm(2017m1)', `=tm(2019m12)')
local lf_r_avg = r(mean)
di `lf_r_avg'
tsline r_lf_rt2 lf_r_rt2 if mo>=`=tm(2019m1)', ///
	yline(`r_lf_avg', lc(black%50) lp(dash)) yline(`lf_r_avg', lc(black%50) lp(dash)) $covid_line $covid_line2 $background ///
	legend(off) xscale(r(`=tm(2019m1)' `=tm(2022m6)')) xlabel(`=tm(2019m1)'(12)$mo_max, format(%tmCY)) xtitle("") ///
	text(0.016   748 "Ret-LF", color(navy)) ///
	text(0.0400  748 "LF-Ret", color(maroon)) ///
	title("Year-to-year flows from labor force to retirement and back", color(black)) ///
	subtitle("Dotted lines reflect 2017-2019 average. Flows expressed as share of 50+ population.", size(small))
	
quietly sum r_lf_rt2 if inrange(mo, `=tm(2017m1)', `=tm(2019m12)')
local r_lf_avg = r(mean)
di `r_lf_avg'
gen r_lf_rt_excess = r_lf_rt2-`r_lf_avg'
quietly sum lf_r_rt2 if inrange(mo, `=tm(2017m1)', `=tm(2019m12)')
local lf_r_avg = r(mean)
di `lf_r_avg'
gen lf_r_rt_excess = lf_r_rt2-`lf_r_avg'

tsline r_lf_rt_excess lf_r_rt_excess if mo>=`=tm(2019m1)', $covid_line $background 

