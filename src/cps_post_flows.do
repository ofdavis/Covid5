clear all 
cd "/users/owen/Covid5"
use data/generated/cps_data.dta, clear 
do src/labels

* rename urate so flow ur can be defined 
rename ur urate

* use this for 2nd months 
xtset cpsidp mo 
gen fmo = f12.mo 
format fmo %tm

* get lwfh, pp from raw cps (move to setup)
frame2 raw, replace 
use data/cps_raw
keep if age>=50 
do src/data_setup_wfh_pp_codes 
format cpsidp %15.0f
gen mo = ym(year,month) 
format mo %tm
keep cpsidp mo lwfh_score pp_score lwfh hpp
tempfile raw 
save "`raw'"
frame change default  
merge 1:1 cpsidp mo using "`raw'"
drop if _merge<3
drop _merge

* ------------------------- load in flow preds ------------------------------- *
local x "a"
local X = strupper("`x'")
di "`X'"
* merge in predictions 
foreach t in er ur nr re ru rn { 
	*local t er 
	local T = strupper("`t'")
	frame2 tmp, replace 
	use data/generated/pred_cps_`T'
	
	keep cpsidp mo p_`T'
	rename *, lower 
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

* define flow indicators 
cap drop er ur nr re ru rn rr l_e l_u l_n l_r
gen er = emp==1 & f12.emp==4
gen ur = emp==2 & f12.emp==4
gen nr = emp==3 & f12.emp==4
gen re = emp==4 & f12.emp==1
gen ru = emp==4 & f12.emp==2
gen rn = emp==4 & f12.emp==3
gen rr = emp==4 & f12.emp==4

* empstat dummies 
gen e = emp==1
gen u = emp==2
gen n = emp==3
gen r = emp==4

* fyr  
gen fyr = year(dofm(fmo))

/* -----------------------------------------------------------------------------
							Transition rates 
						(denom is initial state)
----------------------------------------------------------------------------- */

* ------------------------- collapse by year -----------------------------------
frame copy default test, replace
frame change test
gen college = educ>=3
*drop if fmo<`=tm(2010m1)' 
collapse (sum) er ur nr rr re ru rn e u n r p_er p_ur p_nr p_re p_ru p_rn [fw=wtf12], by(fyr)
drop if fyr==.

tsset fyr 

* some sums 
gen pop = e + u + n + r
gen un = u+n

* create rates of transition to and from retirement 
foreach x in e u n {
	gen `x'r_rt = `x'r/`x'
	gen r`x'_rt = r`x'/r
	gen p_`x'r_rt = p_`x'r/`x'
	gen p_r`x'_rt = p_r`x'/r
	gen diff_`x'r_rt = `x'r_rt-p_`x'r_rt
	gen diff_r`x'_rt = r`x'_rt-p_r`x'_rt
}

cap drop fyr_*
gen fyr_l = fyr+0.2
gen fyr_r = fyr+0.6

* re / re 
local styr 2000
twoway bar diff_er_rt fyr_l if fyr>=`styr', barw(0.4) ///
	|| bar diff_re_rt fyr_r if fyr>=`styr', barw(0.4) ///
	||, xline(2020, lc(black%50) lp(dash)) xmla(`styr'(1)2025, grid tstyle(none)) ///
		xsize(6) ysize(3) ysc(r(-0.005 0.0075)) yla(-0.005(0.0025)0.0075)  name(re, replace)  /// 
		legend(order(1 "Employed to retired" 2 "Retired to employed"))
graph export output/figs/trans_diff_er_re.pdf, replace

* ------------------------- collapse by (byvar), year  -----------------------------------
local byv "college"
frame copy default test, replace
frame change test

gen college = educ>=3

assert `byv'==0 | `byv'==1
drop if fmo<`=tm(2010m1)' 
collapse (sum) er ur nr rr re ru rn e u n r p_er p_ur p_nr p_re p_ru p_rn [fw=wtf12], by(fyr `byv')
drop if fyr==.

xtset `byv' fyr 

* some sums 
gen pop = e + u + n + r
gen un = u+n

* create rates of transition to and from retirement 
foreach x in e u n {
	gen `x'r_rt = `x'r/`x'
	gen r`x'_rt = r`x'/r
	gen p_`x'r_rt = p_`x'r/`x'
	gen p_r`x'_rt = p_r`x'/r
	gen diff_`x'r_rt = `x'r_rt-p_`x'r_rt
	gen diff_r`x'_rt = r`x'_rt-p_r`x'_rt
}

reshape wide er ur nr rr re ru rn e u n r p_er p_ur p_nr p_re p_ru p_rn pop un er_rt re_rt p_er_rt p_re_rt diff_er_rt diff_re_rt ur_rt ru_rt p_ur_rt p_ru_rt diff_ur_rt diff_ru_rt nr_rt rn_rt p_nr_rt p_rn_rt diff_nr_rt diff_rn_rt, i(fyr) j(`byv')

cap drop fyr_*
gen fyr_l = fyr+0.2
gen fyr_r = fyr+0.6

* er/re - non college 
twoway bar diff_er_rt0 fyr_l, barw(0.4) ///
	|| bar diff_re_rt0 fyr_r, barw(0.4) ///
	||, xline(2020, lc(black%50) lp(dash)) xmla(2010(1)2025, grid tstyle(none)) ///
		legend(order(1 "Employed to retired" 2 "Retired to employed") rows(1) pos(6)) name(er, replace) /// 
		xsize(6) ysize(3) ysc(r(-0.01 0.01)) yla(-0.01(0.005)0.01) /// 
		title("Non-college")

* er/re - college 	
twoway bar diff_er_rt1 fyr_l, barw(0.4) ///
	|| bar diff_re_rt1 fyr_r, barw(0.4) ///1
	||, xline(2020, lc(black%50) lp(dash)) xmla(2010(1)2025, grid tstyle(none)) ///
		legend(order(1 "Employed to retired" 2 "Retired to employed") rows(1) pos(6)) name(re, replace) /// 
		xsize(6) ysize(3) ysc(r(-0.01 0.01)) yla(-0.01(0.005)0.01) /// 
		title("College")
		
grc1leg2 er re, rows(2)
graph export output/figs/trans_diff_er_re_edu.pdf, replace


twoway bar diff_ur_rt1 fyr_l, barw(0.4) ///
	|| bar diff_ru_rt1 fyr_r, barw(0.4) //

* ------------------------- binscatter on pp score, ind-occ wage ---------------------------
frame change default 
gen diff = er-p_er

forvalues y=2020/2024 {  // update so that xtitle is phys prox only for 
	local xtitle "  " 
	if `y'==2022 local xtitle "Physical proximity score"
	binscatter2 diff pp_score if fmo>=`=tm(2020m4)' & fyr==`y' & emp==1 [fw=wtf12], name(pp`y', replace) ///
		xtitle("`xtitle'") ytitle("") title(`y')  lc(black) mc(gray) xsize(1.4) ysize(1.4) nq(30) ///
		ysc(r(-0.03 0.03)) yla(-0.03(0.01)0.03)
} 
graph combine pp2020 pp2021 pp2022 pp2023 pp2024, rows(1) imargin(0 0 0 0 ) ///
	xsize(7) ysize(1.4) scale(2)
graph export output/figs/trans_diff_er_scatter_prox.pdf, replace

forvalues y=2020/2024 {  // update so that xtitle is phys prox only for 
	local xtitle "  "
	if `y'==2022 local xtitle "Industry-occupation average wage"
	binscatter2 diff wage_io if fmo>=`=tm(2020m4)' & fyr==`y' & emp==1 [fw=wtf12], name(wage`y', replace) ///
		xtitle("`xtitle'") ytitle("") title(`y')  lc(black) mc(gray) xsize(1.4) ysize(1.4) nq(30) ///
		ysc(r(-0.02 0.03)) yla(-0.02(0.01)0.03)
} 
graph combine wage2020 wage2021 wage2022 wage2023 wage2024, rows(1) imargin(0 0 0 0 ) ///
	xsize(7) ysize(1.4) scale(2)
graph export output/figs/trans_diff_er_scatter_wage.pdf, replace



/* -----------------------------------------------------------------------------
									Flows
----------------------------------------------------------------------------- */
* ------------------------- collapse overall, year -----------------------------------
frame copy default coll, replace
frame change coll
collapse (sum) er ur nr rr re ru rn e u n r p_er p_ur p_nr p_re p_ru p_rn [fw=wtf12], by(fmo)
drop if fmo==.

tsset fmo 

* some sums 
gen pop = e + u + n + r
gen un = u+n

* transition rate -- use 50+ pop as denominator >>this is the one used in the graph
foreach x in e u n {
	gen `x'r_rt = `x'r/pop
	gen r`x'_rt = r`x'/pop
	gen p_`x'r_rt = p_`x'r/pop
	gen p_r`x'_rt = p_r`x'/pop
}

* combine lf components 
gen lr_rt = er_rt + ur_rt
gen rl_rt = re_rt + ru_rt
gen p_lr_rt = p_er_rt + p_ur_rt
gen p_rl_rt = p_re_rt + p_ru_rt

* combine All non-ret components 
gen ar_rt = er_rt + ur_rt + nr_rt
gen ra_rt = re_rt + ru_rt + rn_rt
gen p_ar_rt = p_er_rt + p_ur_rt + p_nr_rt
gen p_ra_rt = p_re_rt + p_ru_rt + p_rn_rt

* create diffs and smooth 
foreach var in er re ur ru nr rn { 
	gen diff_`var'_rt = `var'_rt-p_`var'_rt
	*tssmooth ma `var'_rt_ma = `var'_rt, window(11 1) 
	*tssmooth ma p_`var'_rt_ma = p_`var'_rt, window(11 1) 
	tssmooth ma diff_`var'_rt_ma = diff_`var'_rt, window(11 1) 
}

* plot each pair of transitions w/ predictions 
foreach x in e u n {
	local mo `=tm(2022m1)'
	if "`x'"=="e" local varname "employed"
	if "`x'"=="u" local varname "unemployed"
	if "`x'"=="n" local varname "not-in-LF"
	qui sum `x'r_rt if fmo==`=tm(2024m12)'
	local `x'r_y = r(mean)+0.001
	qui sum r`x'_rt if fmo==`=tm(2024m12)'
	local r`x'_y = r(mean)-0.001
	
	colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
	twoway tsline `x'r_rt p_`x'r_rt if fmo>`=tm(2018m1)', lc("`r(p1)'" "`r(p1)'%50") lp(solid dash) ///  ///
		|| tsline r`x'_rt p_r`x'_rt if fmo>`=tm(2018m1)', lc("`r(p2)'" "`r(p2)'%50") lp(solid dash) /// 
		||, xtitle("") xla(,format(%tmCY)) $covid_line xsize(3) ysize(2.5) ysc(r(0)) yla(0(0.01)0.04)  /// 
		text(`r`x'_y' `=tm(2025m1)' "Out of ret.",   placement(e) j(left) size(small)) /// 
		text(``x'r_y' `=tm(2025m1)' "Into ret.", placement(e) j(left) size(small)) /// 
		legend(order(1 "Actual transition" 2 "Predicted") pos(6) rows(1)) ///
		title("Retirement transitions involving `varname'", size(medsmall)) name(`x'_graph,replace)
}
grc1leg2 e_graph u_graph n_graph, imargin(0 15 1 1)
graph export output/figs/flows.pdf, replace


* ------------------------- collapse overall, year -----------------------------------
frame copy default coll, replace
frame change coll
drop if inrange(fmo,`=tm(2020m1)',`=tm(2020m3)')
*drop if mo<`=tm(2010m1)'
collapse (sum) er ur nr rr re ru rn e u n r p_er p_ur p_nr p_re p_ru p_rn [fw=wtf12], by(fyr)
drop if fyr==.

tsset fyr 

* some sums 
gen pop = e + u + n + r
gen un = u+n

* transition rate -- use 50+ pop as denominator >>this is the one used in the graph
foreach x in e u n {
	gen `x'r_rt = `x'r/pop
	gen r`x'_rt = r`x'/pop
	gen p_`x'r_rt = p_`x'r/pop
	gen p_r`x'_rt = p_r`x'/pop
}

* altnerate ur prediction for 2020+ --- treat ur_rt>avg(2018-2019) as pred
sum ur_rt if inrange(fyr,2018, 2019)
gen p_ur_rt_alt = p_ur_rt
replace p_ur_rt_alt = r(mean) if fyr>=2020

* combine lf components 
gen lr_rt = er_rt + ur_rt
gen rl_rt = re_rt + ru_rt
gen p_lr_rt = p_er_rt + p_ur_rt
gen p_rl_rt = p_re_rt + p_ru_rt

gen p_lr_rt_alt = p_er_rt + p_ur_rt_alt

* combine All non-ret components 
gen ar_rt = er_rt + ur_rt + nr_rt
gen ra_rt = re_rt + ru_rt + rn_rt
gen p_ar_rt = p_er_rt + p_ur_rt + p_nr_rt
gen p_ra_rt = p_re_rt + p_ru_rt + p_rn_rt

gen p_ar_rt_alt = p_er_rt + p_ur_rt_alt + p_nr_rt

* create diffs 
foreach var in er_rt re_rt ur_rt ru_rt nr_rt rn_rt lr_rt rl_rt {
	gen diff_`var' = `var'-p_`var'
}
gen diff_rl_rt_ = -diff_rl_rt 
gen diff_ur_rt_alt = ur_rt - p_ur_rt_alt 
gen diff_lr_rt_alt = lr_rt - p_lr_rt_alt

gen total_rt = diff_rl_rt_ + diff_lr_rt
gen total_rt_alt = diff_rl_rt_ + diff_lr_rt_alt

* graph -- need to specify how stack looks depending on signs of total and constituents
cap drop fyr_
gen fyr_ = fyr+0.5 // for xtick offset 

colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway bar diff_lr_rt fyr_, color(black) /// 
	|| rbar diff_lr_rt total_rt fyr_ if (diff_lr_rt>0 & diff_rl_rt_>0), color(gray)	 /// both positive
	|| rbar total_rt diff_lr_rt fyr_ if (diff_lr_rt<0 & diff_rl_rt_<0), color(gray)	/// both negative 
	|| bar  diff_rl_rt_ fyr_ if (diff_lr_rt>0 & diff_rl_rt_<0) | (diff_lr_rt<0 & diff_rl_rt_>0), color(gray)	/// 
	|| line total_rt fyr_, lw(0.5) lc("`r(p3)'") ///
	|| scatter total_rt fyr_, ms(C) mc("`r(p3)'") ///
	||, legend(order(1 "Retirements" "(above trend)" 2 "Unretirements" "(below trend)" ///
					 5 "Total contribution" "to retired share")) /// 
	xtitle("")  xmla(2000(1)2025, grid tstyle(none)) $covid_line_yr xsize(6)

* graph bar stack 
graph bar diff_er_rt diff_ur_rt_alt diff_rl_rt_, over(fyr, label ) stack ///
	legend(order(1 "Employed to retired" 2 "Unemployed to retired" 3 "Retired to LF (inverted)")) ///
	xsize(6) name(stack, replace)

* ------------------------------ manual stack --------------------------------- * 
cap drop x* 
cap drop total 
cap drop high 
cap drop low  
gen x1 = diff_er_rt 
gen x2 = diff_ur_rt_alt 
gen x3 = diff_rl_rt_

* Suppose your data have variables: year, x1, x2, x3

* (Optional) Create a total variable for the overlaid line plot:
gen total = x1 + x2 + x3

* ------------- 
* For x1: simply, if x1 is positive, draw from 0 up; if negative, from x1 up to 0.
gen 	x1_bot = . 
gen 	x1_top = . 
replace x1_bot = 0  if x1>0 // x1>0
replace x1_top = x1 if x1>0
replace x1_bot = x1 if x1<0 // x1<0
replace x1_top = 0  if x1<0 

* x2 ------------
gen 	x2_bot = . 
gen 	x2_top = . 
replace x2_bot = x1_top 	 if x1>0 & x2>0 // both positive 
replace x2_top = x1_top + x2 if x1>0 & x2>0 
replace x2_bot = x2	 	 	 if x1>0 & x2<0 // x1 positive, x2 negative 
replace x2_top = 0 			 if x1>0 & x2<0 
replace x2_bot = 0	 	 	 if x1<0 & x2>0 // x1 negative, x2 postiive 
replace x2_top = x2 		 if x1<0 & x2>0 
replace x2_bot = x2	+ x1_bot if x1<0 & x2<0 // both negative 
replace x2_top = x1_bot  	 if x1<0 & x2<0 

* x3 ------------
* get highest and lowest points 
egen high = rowmax(x1_top x2_top)
egen low  = rowmin(x1_bot x2_bot)
gen 	x3_bot = . 
gen 	x3_top = . 
replace x3_bot = high 		if x3>0 // x3 positive 
replace x3_top = high+x3    if x3>0 
replace x3_bot = low +x3    if x3<0 // x3 negative 
replace x3_top = low        if x3<0

* plot sequence 
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway rbar x1_bot x1_top fyr_,  color("`r(p1)'")    ///
    || rbar x2_bot x2_top fyr_,  color(white%0) lc(white%0)    ///
    ||,  legend(order(1 "Employed to retired     " "(excess)")  ///
				    size(small))  ///
	xsize(6) xtitle("")  ytitle("") $covid_line_yr ysc(r(-0.004 0.005)) ///
	name(flows_total_1, replace ) 
graph export output/figs/flows_total_1.pdf, replace 
	
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway rbar x1_bot x1_top fyr_,  color("`r(p1)'")    ///
    || rbar x2_bot x2_top fyr_,  color("`r(p3)'")    ///
    || , legend(order(1 "Employed to retired     " "(excess)"  ///
				   2 "Unemployed to retired" "(excess*)") size(small))  ///
	xsize(6) xtitle("") $covid_line_yr ysc(r(-0.004 0.005)) ///
	name(flows_total_2, replace ) 
graph export output/figs/flows_total_2.pdf, replace
	
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway rbar x1_bot x1_top fyr_,  color("`r(p1)'")    ///
    || rbar x2_bot x2_top fyr_,  color("`r(p3)'")    ///
    || rbar x3_bot x3_top fyr_,  color("`r(p2)'")    ///
    || , legend(order(1 "Employed to retired     " "(excess)"  ///
				   2 "Unemployed to retired" "(excess*)" /// 
				   3 "Retired to labor force" "(excess, inverted)" ) size(small))  ///
	xsize(6) xtitle("") $covid_line_yr ysc(r(-0.004 0.005)) ///
	name(flows_total_3, replace ) 
graph export output/figs/flows_total_3.pdf, replace
	
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway rbar x1_bot x1_top fyr_,  color("`r(p1)'")    ///
    || rbar x2_bot x2_top fyr_,  color("`r(p3)'")    ///
    || rbar x3_bot x3_top fyr_,  color("`r(p2)'")    ///
    || line total fyr_, lwidth(0.5) lcolor("`r(p4)'")   ///
    || scatter total fyr_, ms(S) mcolor("`r(p4)'")   ///
    || , legend(order(1 "Employed to retired     " "(excess)"  ///
				   2 "Unemployed to retired" "(excess*)" ///
				   3 "Retired to labor force" "(excess, inverted)" /// 
				   4  "Total contribution" "to retired share") size(small))  ///
	xsize(6) xtitle("") $covid_line_yr ysc(r(-0.004 0.005)) ///
	name(flows_total, replace ) 
graph export output/figs/flows_total.pdf, replace
