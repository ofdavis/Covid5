frame change default
cd "/users/owen/Covid5"
use data/covid_long.dta, clear
rename ur urate

qui sum mo 
global mo_max = r(max)
global covid_line "xline(`=tm(2020m4)', lp(dot) lc(black))"
global covid_line2 "xline(`=tm(2021m3)', lp(dot) lc(black))"

*create indicators for flows 
gen er = emp==1 & f12.emp==4 if emp==1

* gen agegrp=(age>=60) + (age>=65) + (age>=70)
* label define agegrp 0 "50-59" 1 "60-64" 2 "65-69" 3 "70+"
* label values agegrp agegrp 

frame copy default coll, replace 
frame change coll 
collapse (mean) er [fw=wtf12], by(mo sex)
xtset sex mo 
replace mo = mo+12
replace er = er*100 

gen period = inrange(mo,`=tm(2020m4)',`=tm(2021m3)') if inrange(mo,`=tm(2018m4)',`=tm(2021m3)')

forvalues p=0/1 { 
forvalues s=0/1 {
	qui sum er if period==`p' & sex==`s'
	gen m`p'`s'=r(mean) if period==`p' 
	gen m_`p'`s' = r(mean) if mo==`=tm(2021m4)'
}
}

gen period1 = inrange(mo,`=tm(2018m4)',`=tm(2021m5)')

qui sum m00
local m00 = r(mean)
qui sum m01
local m01 = r(mean)
qui sum m10
local m10 = r(mean)
qui sum m11
local m11 = r(mean)
local m0chg = string(`m10'-`m00', "%9.1f")
local m1chg = string(`m11'-`m01', "%9.1f")
di "`m0chg' `m1chg'"
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway tsline er if sex==0 & period<., lp(solid) lc("`r(p1)'%50") ///
	|| tsline er if sex==1 & period<., lp(solid) lc("`r(p3)'%50") ///
	|| tsline m00 m10 m01 m11 if period1==1, lw(0.6 0.6 0.6 0.6) lc("`r(p1)'" "`r(p1)'" "`r(p3)'" "`r(p3)'") ///
	|| rcap m_00 m_10 mo if period1==1, lc("`r(p1)'") /// 
	|| rcap m_01 m_11 mo if period1==1, lc("`r(p3)'") /// 
	||, xsc(r(`=tm(2018m4)' `=tm(2021m3)')) xla(`=tm(2018m1)'(12)`=tm(2021m1)', format(%tmCY)) ///
	yla(5 "5%" 6 "6%" 7 "7%" 8 "8%" 9 "9%") /// 
	$covid_line xtitle("") ///
	legend(order(3 "Men" 5 "Women" 7 "Change, men" 8 "Change, women" )) ///
	title("Percent retired among those employed 1 year prior", size(medium)) ///
	xsize(6) /// 
	text(6.6 738 "=`m0chg'%"  ) ///
	text(7.4 738 "=`m1chg'%", c("`r(p3)'")  ) //
graph export "output/figs/diff_demonstration.pdf", replace
