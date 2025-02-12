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
		xsize(6) ysize(3) ysc(r(-0.01 0.01)) yla(-0.01(0.005)0.01)

* er/re - college 	
twoway bar diff_er_rt1 fyr_l, barw(0.4) ///
	|| bar diff_re_rt1 fyr_r, barw(0.4) ///1
	||, xline(2020, lc(black%50) lp(dash)) xmla(2010(1)2025, grid tstyle(none)) ///
		legend(order(1 "Employed to retired" 2 "Retired to employed") rows(1) pos(6)) name(re, replace) /// 
		xsize(6) ysize(3) ysc(r(-0.01 0.01)) yla(-0.01(0.005)0.01)
		
grc1leg2 er re, rows(2)


* ------------------------- binscatter on pp score, ind-occ wage ---------------------------
frame change default 
gen diff = er-p_er

forvalues y=2020/2024 {  // update so that xtitle is phys prox only for 
	local xtitle "  " 
	if `y'==2022 local xtitle "Physical proximity score"
	binscatter2 diff pp_score if fmo>=`=tm(2020m4)' & fyr==`y' & emp==1 [fw=wtf12], name(pp`y', replace) ///
		xtitle("`xtitle'") ytitle("") title(`y')  lc(black) mc(gray) xsize(1.4) ysize(1.4) nq(30) ///
		ysc(r(-0.02 0.03)) yla(-0.02(0.01)0.03)
} 
graph combine pp2020 pp2021 pp2022 pp2023 pp2024, rows(1) imargin(0 0 0 0 ) ///
	xsize(7) ysize(1.4) scale(2)

forvalues y=2020/2024 {  // update so that xtitle is phys prox only for 
	local xtitle "  "
	if `y'==2022 local xtitle "Industry-occupation average wage"
	binscatter2 diff wage_io if fmo>=`=tm(2020m4)' & fyr==`y' & emp==1 [fw=wtf12], name(wage`y', replace) ///
		xtitle("`xtitle'") ytitle("") title(`y')  lc(black) mc(gray) xsize(1.4) ysize(1.4) nq(30) ///
		ysc(r(-0.02 0.03)) yla(-0.02(0.01)0.03)
} 
graph combine wage2020 wage2021 wage2022 wage2023 wage2024, rows(1) imargin(0 0 0 0 ) ///
	xsize(7) ysize(1.4) scale(2)



/* -----------------------------------------------------------------------------
							Transition rates 
						(denom is initial state)
----------------------------------------------------------------------------- */

* ------------------------- collapse overall -----------------------------------
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
gen rl_rt2_diff_ = -rl_rt2_diff 

gen total_rt2 = rl_rt2_diff_ + lr_rt2_diff

* graph -- need to specify how stack looks depending on signs of total and constituents
cap drop fyr_
gen fyr_ = fyr+0.5 // for xtick offset 

colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway bar lr_rt2_diff fyr_, color(black) /// 
	|| rbar lr_rt2_diff total_rt2 fyr_ if (lr_rt2_diff>0 & rl_rt2_diff_>0), color(gray)	 /// both positive
	|| rbar total_rt2 lr_rt2_diff fyr_ if (lr_rt2_diff<0 & rl_rt2_diff_<0), color(gray)	/// both negative 
	|| bar  rl_rt2_diff_ fyr_ if (lr_rt2_diff>0 & rl_rt2_diff_<0) | (lr_rt2_diff<0 & rl_rt2_diff_>0), color(gray)	/// 
	|| line total_rt2 fyr_, lw(0.5) lc("`r(p3)'") ///
	|| scatter total_rt2 fyr_, ms(C) mc("`r(p3)'") ///
	||, legend(order(1 "Retirements" "(excess)" 2 "Unretirements" "()" 5 "Total contribution" "to retired share")) /// 
	xtitle("")  xmla(2000(1)2025, grid tstyle(none)) $covid_line_yr xsize(6)


























