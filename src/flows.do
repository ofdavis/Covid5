frame change default
cd "/users/owen/Covid5"
use data/covid_long.dta, clear
rename ur urate
	
qui sum mo 
global mo_max = r(max)
global covid_line "xline(`=tm(2020m4)', lp(dot) lc(black))"
global covid_line2 "xline(`=tm(2021m3)', lp(dot) lc(black))"

*create indicators for flows 
cap drop er ur nr re ru rn rr l_e l_u l_n l_r
gen er = emp==1 & f12.emp==4
gen ur = emp==2 & f12.emp==4
gen nr = emp==3 & f12.emp==4
gen re = emp==4 & f12.emp==1
gen ru = emp==4 & f12.emp==2
gen rn = emp==4 & f12.emp==3
gen rr = emp==4 & f12.emp==4

gen fmo = f12.mo 
format fmo %tm

* create indicators for labor force totals 
gen e = emp==1
gen u = emp==2
gen n = emp==3
gen r = emp==4

* merge in predictions 
foreach t in er ur nr re ru rn { 
	*local t er 
	frame2 tmp, replace 
	use data/generated/trans_`t'_nn 
	keep cpsidp mo py2_nn 
	rename py2_nn p_`t'
	keep if mo<.
	rename mo fmo
	tempfile tmp
	save "`tmp'"
	frame change default 
	di "====merging in `t' predictions=========="
	merge m:1 cpsidp fmo using "`tmp'"
	xtset cpsidp mo
	if "`t'"=="er" {
		assert f12.cpsidp<. & emp==1 if _merge==3
		assert f12.cpsidp==. | emp!=1 if _merge==1
		assert _merge!=2
	}
	if "`t'"=="ur" { 
		assert f12.cpsidp<. & emp==2 if _merge==3
		assert f12.cpsidp==. | emp!=2 if _merge==1
		assert _merge!=2
	}
	if "`t'"=="nr" { 
		assert f12.cpsidp<. & emp==3 if _merge==3
		assert f12.cpsidp==. | emp!=3 if _merge==1
		assert _merge!=2
	}
	if inlist("`t'","re","ru","rn") { 
		assert f12.cpsidp<. & emp==4 if _merge==3
		assert f12.cpsidp==. | emp!=4 if _merge==1
		assert _merge!=2
	}
	drop _merge
}

frame copy default flows_rates_yr_50, replace
frame change flows_rates_yr_50
drop if mo<`=tm(2010m1)'
collapse (sum) er ur nr rr re ru rn e u n r p_er p_ur p_nr p_re p_ru p_rn [fw=wtf12], by(fmo) //[fw=yearwt] <-- need to redo 
drop if fmo==.

tsset fmo 

* some sums 
gen pop = e + u + n + r
gen un = u+n

* create rates of transition to and from retirement 
foreach x in e u n {
	gen `x'r_rt = `x'r/`x'
	gen r`x'_rt = r`x'/r
	gen p_`x'r_rt = p_`x'r/`x'
	gen p_r`x'_rt = p_r`x'/r
}

* alternative rate var -- use 50+ pop as denominator >>this is the one used in the graph
foreach x in e u n {
	gen `x'r_rt2 = `x'r/pop
	gen r`x'_rt2 = r`x'/pop
	gen p_`x'r_rt2 = p_`x'r/pop
	gen p_r`x'_rt2 = p_r`x'/pop
}

* combine lf components 
gen lr_rt2 = er_rt2 + ur_rt2
gen rl_rt2 = re_rt2 + ru_rt2
gen p_lr_rt2 = p_er_rt2 + p_ur_rt2
gen p_rl_rt2 = p_re_rt2 + p_ru_rt2

* combine All non-ret components 
gen ar_rt2 = er_rt2 + ur_rt2 + nr_rt2
gen ra_rt2 = re_rt2 + ru_rt2 + rn_rt2
gen p_ar_rt2 = p_er_rt2 + p_ur_rt2 + p_nr_rt2
gen p_ra_rt2 = p_re_rt2 + p_ru_rt2 + p_rn_rt2

* create diffs 
foreach var in er_rt re_rt ur_rt ru_rt nr_rt rn_rt er_rt2 re_rt2 ur_rt2 ru_rt2 nr_rt2 rn_rt2 lr_rt2 rl_rt2 {
	gen `var'_diff = `var'-p_`var'
}

* moving averages 
foreach var of varlist _all { 
	if "`var'"!="fmo" { 
		qui tssmooth ma `var'_ma = `var', window(11 1) 
	}
}

/* fig with all ret flows 
frame change flows_rates_yr_50
tsline er_rt2 re_rt2 ur_rt2 ru_rt2 nr_rt2 rn_rt2, ///
	lc(navy navy maroon maroon forest_green forest_green) lp(solid shortdash solid shortdash solid shortdash) ///
	name(`frm'_2, replace) $covid_line $covid_line2 xtitle("") legend(off) ///
	text(0.034 755 "Emp-Ret", color(navy)) text(0.029  755 "NILF-Ret", color(forest_green))  text(0.019  755 "Ret-NILF", color(forest_green)) ///
	text(0.014 755 "Ret-Emp", color(navy)) text(0.0045 756 "Unem-Ret", color(maroon))  		 text(0.0015 756 "Ret-Unem", color(maroon)) ///
	xscale(r(`=tm(2011m1)' `=tm(2023m3)')) xlabel(`=tm(2011m1)'(12)$mo_max, format(%tmCY))
*/ 
	

	
* ----- flows to and from LF and ret -----
foreach var in lr_rt2 rl_rt2 { 
	qui sum `var' if inrange(fmo,`=tm(2017m1)',`=tm(2019m12)')
	local mean_`var' = r(mean)
} 
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
tsline lr_rt2 rl_rt2 if fmo>=`=tm(2018m1)', ///
	///*lc("`r(p1)'" "`r(p1)'" "`r(p2)'" "`r(p2)'" )*/ lp(solid shortdash solid shortdash) ///
	name(`frm'_2, replace) $covid_line $covid_line2 xtitle("") ///
	legend(order(1 "Labor force to retired"  - " " - " " - " " - " " - " " - " " - " " - " " - " " - " " ///
			     2 "Retired to labor force"- " " - " " )) ///
	xscale(r(`=tm(2018m1)' `=tm(2024m1)')) xlabel(`=tm(2018m1)'(12)`=tm(2024m1)', format(%tmCY)) ///
	yline( `mean_lr_rt2', lp(dash) lc("`r(p1)'%75")) /// 
	yline( `mean_rl_rt2', lp(dash) lc("`r(p2)'%75")) ///
	xsize(7)
	
graph export output/figs/flows_lfr.pdf, replace


* ----- flows to and from E/U and ret -----
foreach var in er_rt2 re_rt2 ur_rt2 ru_rt2 { 
	qui sum `var' if inrange(fmo,`=tm(2017m1)',`=tm(2019m12)')
	local mean_`var' = r(mean)
} 
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
tsline er_rt2 re_rt2 ur_rt2 ru_rt2 if fmo>=`=tm(2018m1)', ///
	///*lc("`r(p1)'" "`r(p1)'" "`r(p2)'" "`r(p2)'" )*/ lp(solid shortdash solid shortdash) ///
	name(`frm'_2, replace) $covid_line $covid_line2 xtitle("") ///
	legend(order(1 "Employed to retired" - " " - " " - " " - " " - " " - " " - " " ///
				 2 "Retired to employed" - " " - " " - " " ///  
				 3 "Unemployed to retired" 4 "Retired to unemployed")) ///
	xscale(r(`=tm(2018m1)' `=tm(2024m1)')) xlabel(`=tm(2018m1)'(12)`=tm(2024m1)', format(%tmCY)) ///
	yline( `mean_er_rt2', lp(dash) lc("`r(p1)'%75")) /// 
	yline( `mean_re_rt2', lp(dash) lc("`r(p2)'%75")) /// 
	yline( `mean_ur_rt2', lp(dash) lc("`r(p3)'%75")) /// 
	yline( `mean_ru_rt2', lp(dash) lc("`r(p4)'%75")) ///
	xsize(7)

graph export output/figs/flows_eur.pdf, replace




* ----------combined LF-ret with ML predictions  ----------
* combined LF-ret 
twoway	tsline lr_rt2 rl_rt2  if fmo>=`=tm(2018m1)', lc(black gray) lp(dot dot) ///
	||  tsline lr_rt2_ma rl_rt2_ma  if fmo>=`=tm(2018m1)', lc(black gray) ///
	||  tsline p_lr_rt2_ma  p_rl_rt2_ma  if fmo>=`=tm(2018m1)', lc(black gray) lp(dash dash) ///
	||, ///	yline(`rl_avg', lc(black%50) lp(dash)) yline(`lr_avg', lc(black%50) lp(dash)) ///
	$covid_line $covid_line2 $background ///
	legend(order(- "{bf:Labor force to retired}" 1 "Raw" 3 "Smoothed" 5 "Predicted," "smoothed" ///
				 - " " - " " - " " - " " - " " - " " ///
				 - "{bf:Retired to labor force}" 2 "Raw" 4 "Smoothed" 6 "Predicted," "smoothed") pos(3))  ///
	xscale(r(`=tm(2018m1)' `=tm(2024m1)')) xlabel(`=tm(2018m1)'(12)`=tm(2024m1)', format(%tmCY)) xtitle("") ///
	xsize(7)
	
graph export output/figs/flows_lfr_pred.pdf, replace




* er with predictions 
tsline er_rt2_ma p_er_rt2_ma re_rt2_ma p_re_rt2_ma if inrange(fmo, `=tm(2019m1)',`=tm(2024m12)')/*unr_rt2 p_unr_rt2*/, ///
	legend(off) $covid_line $covid_line2

* lf r with predictions 
tsline lr_rt2_ma p_lr_rt2_ma rl_rt2_ma p_rl_rt2_ma,  ///
	legend(off) ysc()
