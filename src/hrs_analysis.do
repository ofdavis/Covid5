
clear all 
cd "/Users/owen/Covid5/Data/HRS/raw/sects/"

* -------------------------------------------------------------------------- * 
* ------------------------ 	regression with covid -------------------------- * 
* -------------------------------------------------------------------------- * 
* goal: among those employed in wave 13, retirement outcomes pre-post covid 
* ironically, need wide data for this! 
frame change default 
use hrs_monthly_temp, clear 

* indicate if ever unem, nilf during a wave 
gen unem = lbrff==2 
gen nilf = lbrff==5
bys hhidpn wave: egen unem_wv=max(unem)
bys hhidpn wave: egen nilf_wv=max(nilf)

keep if mo==iwdate
drop if dead
gen retired=lbrff==3

* create 13-only vars to merge back 
keep hhidpn wave incovid mo lbrff empchg mstatf hhid pn wgtr gender race foreign educ nurshm cendiv urbrur hh_numch hh_numad hh_rent hh_hhres hh_child hh_achck hh_ahous hh_amort hh_atoth hh_atotf hh_icap btwjob sayret shlt hltc jhours slfemp samemp jcten jcoccc jcindc wgihr wgiwk fsize union jcpen age jhours_flag slfemp_flag jcten_flag jcoccc_flag jcindc_flag wgihr_flag wgiwk_flag fsize_flag union_flag jcpen_flag ft lachck lahous lamort latoth latotf licap any_adult any_child sp_lbrff sp_dead sp_nurshm sp_shlt sp_age sp_ft covid retired covid_any wgtr_c covwk covwk_affect covwk_stop covwk_event covwk_findnew covwk_howaffect covwk_risk covwk_hard covwk_wfh covwk_ownbiz covwk_ownaffect covwk_ownclose covwk_ownpermclose covid_concern died* nilf_wv unem_wv

reshape wide mo lbrff empchg mstatf wgtr nurshm cendiv urbrur hh_numch hh_numad hh_rent hh_hhres hh_child hh_achck hh_ahous hh_amort hh_atoth hh_atotf hh_icap btwjob sayret shlt hltc jhours slfemp samemp jcten jcoccc jcindc wgihr wgiwk fsize union jcpen age jhours_flag slfemp_flag jcten_flag jcoccc_flag jcindc_flag wgihr_flag wgiwk_flag fsize_flag union_flag jcpen_flag ft lachck lahous lamort latoth latotf licap any_adult any_child sp_lbrff sp_dead sp_nurshm sp_shlt sp_age sp_ft covid retired covid_any wgtr_c covwk covwk_affect covwk_stop covwk_event covwk_findnew covwk_howaffect covwk_risk covwk_hard covwk_wfh covwk_ownbiz covwk_ownaffect covwk_ownclose covwk_ownpermclose covid_concern  nilf_wv unem_wv, i(hhidpn) j(wave)

foreach var in covwk covwk_affect covwk_stop covwk_event covwk_findnew covwk_howaffect covwk_risk covwk_hard covwk_wfh covwk_ownbiz covwk_ownaffect covwk_ownclose covwk_ownpermclose covid_concern { 
	drop `var'13 `var'14 `var'16
	rename `var'15 `var'
	replace `var'=0 if `var'==.
}
gen covwk_lose = covwk_event==1
gen covwk_layoff = covwk_event==2
gen covwk_quit = covwk_event==3
gen covwk_retire = covwk_event==5
replace covid_concern =  covid_concern/10

replace sp_lbrff13=0 if sp_lbrff13==.
replace sp_lbrff14=0 if sp_lbrff14==.
replace sp_age13=0 if sp_age13==.
replace sp_age14=0 if sp_age14==.

gen retnilf16 = inrange(lbrff16,3,5)

* among those working in 2016/2018 and in covid sample, predict retirement in 2022 
gen retired_any = retired15==1 | retired16==1
gen sayret_any = sayret15==1 | sayret16==1
gen samp = lbrff14==1 & incovid==1 & covwk==1 & ((wgtr_c15>0 & wgtr_c15<.) | (wgtr_c16>0 & wgtr_c16<.))

gen covid_diag = max(covid_any15,covid_any16)

* month of final interview 
gen mo_final = mo16 
replace mo_final = mo15 if mo16==.
format mo_final %tm

* kitchen sink reg 
logit sayret_any c.age14##c.age14 i.mstatf14 i.mstatf14#c.sp_age14 i.sp_lbrff14 i.shlt14 ///
	hh_hhres14 hh_ahous14 hh_atotf14 hh_rent14 ///
	slfemp14 jhours14 wgiwk14 jcten14  /// 
	i.cendiv14 i.educ gender i.race foreign i.urbrur14 i.jcoccc14 i.jcindc14 /// 
	i.covwk_lose  covwk_wfh covwk_hard covid_concern covid_diag ///  
		[pw=wgtr_c14] if samp==1, vce(rob) or
est sto r2_allvars 

* basic reg with all covid vars (using same sample as before)
logit sayret_any c.age14##c.age14 i.educ gender i.race ///
	covwk_lose covwk_wfh covwk_hard covid_concern covid_diag  ///  
	[pw=wgtr_c14] if _est_r2_allvars==1,  vce(rob)
	est sto r1_allvars
	
* run thru each covid var separately 
local rtext ""
foreach var in covwk_lose covwk_wfh covwk_hard covid_concern covid_diag {
	logit sayret_any c.age14##c.age14 ///
	i.educ gender i.race /// 
		 `var' ///  
		[pw=wgtr_c14] if _est_r2_allvars==1,  vce(rob)
	est sto r1_`var'

	logit sayret_any c.age14##c.age14 i.mstatf14 i.mstatf14#c.sp_age14 i.sp_lbrff14 i.shlt14 ///
		hh_hhres14 hh_ahous14 hh_atotf14 hh_rent14 ///
		slfemp14  jhours14 wgiwk14 jcten14  /// 
		i.cendiv14 i.educ gender foreign i.urbrur14 i.race i.jcoccc14 i.jcindc14  /// 
		`var'  /// 
			[pw=wgtr_c14] if _est_r2_allvars==1, ///
			vce(rob)
			
	est sto r2_`var'
	
	local rtext = "`rtext' " + "r1_`var' " + "r2_`var'"
}
di "`rtext'"
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
coefplot (r1_allvars,       mcolor(black) ciopts(color(black))) ///
		 (r2_allvars,       mcolor(gray) ciopts(color(gray))) ///
		 (r1_covwk_lose,    mcolor("`r(p3)'") ciopts(color("`r(p3)'"))) ///
		 (r2_covwk_lose,    mcolor("`r(p4)'") ciopts(color("`r(p4)'"))) /// 
		 (r1_covwk_wfh,     mcolor("`r(p3)'") ciopts(color("`r(p3)'"))) ///
		 (r2_covwk_wfh,     mcolor("`r(p4)'") ciopts(color("`r(p4)'"))) ///
		 (r1_covwk_hard,    mcolor("`r(p3)'") ciopts(color("`r(p3)'"))) /// 
		 (r2_covwk_hard,    mcolor("`r(p4)'") ciopts(color("`r(p4)'"))) ///
		 (r1_covid_concern, mcolor("`r(p3)'") ciopts(color("`r(p3)'"))) /// 
		 (r2_covid_concern, mcolor("`r(p4)'") ciopts(color("`r(p4)'"))) /// 
		 (r1_covid_diag, mcolor("`r(p3)'") ciopts(color("`r(p3)'"))) /// 
		 (r2_covid_diag, mcolor("`r(p4)'") ciopts(color("`r(p4)'"))) /// 
	 , keep(covwk_lose covwk_wfh covwk_hard covid_concern covid_diag) /// 
	 legend(order(2 "All Covid variables," "simple controls" /// 
				  4 "All Covid variables," "full controls" ///   
				  6 "Single Covid variable," "simple controls" ///  
				  8 "Single Covid variable," "full controls"    ) size(medsmall) pos(3)) /// 
	 coeflabel(covwk_lose     = "Lost job during Covid" /// 
			   covwk_wfh      = "Switched to remote" /// 
			   covwk_hard     = "Covid made work harder" /// 
			   covid_concern  = "Level of Covid concern" /// 
			   covid_diag    = "Covid diagnosis", labsize(medsmall)) ///
	xline(0, lc(black) lp(dash)) xsize(7.5)
graph export "/Users/owen/Covid5/output/figs/hrs_covidvar_regs.pdf", replace




* -------------------------------------------------------------------------- * 
* ------------------------ 	descriptive covid vars ------------------------- * 
* -------------------------------------------------------------------------- * 
* goal: among those employed in wave 13, retirement outcomes pre-post covid 
* ironically, need wide data for this! 
frame change default 
use hrs_monthly_temp, clear 
keep if mo==iwdate & wave==15
keep if wgtr_c<. & wgtr_c>0
tab mo 

* ----------- ladder 1: work, work affected stop work, 
* how many working at covid outbreak? 
table covwk [fw=round(wgtr_c)], stat(frequency) stat(sumw) stat(rawpercent) stat(percent)

* how many had work affected?
table covwk_affect if covwk==1 [fw=round(wgtr_c)], stat(frequency) stat(sumw) stat(rawpercent) stat(percent)

* work stopped?
table covwk_stop if covwk_affect==1 [fw=round(wgtr_c)], stat(frequency) stat(sumw) stat(rawpercent) stat(percent)

* why stopped? 
table covwk_event if covwk_stop==1 [fw=round(wgtr_c)], stat(frequency) stat(sumw) stat(rawpercent) stat(percent) miss

* new job? 
table covwk_findnew if covwk_stop==1 [fw=round(wgtr_c)], stat(frequency) stat(sumw) stat(rawpercent) stat(percent) miss



* ----------- ladder 2: work, work affected but not stopped , how affected 
* how many working at covid outbreak? 
table covwk [fw=round(wgtr_c)], stat(frequency) stat(sumw) stat(rawpercent) stat(percent)

* how many had work affected but not stopped?
table covwk_affect if covwk==1 & covwk_stop!=1 [fw=round(wgtr_c)], stat(frequency) stat(sumw) stat(rawpercent) stat(percent)

* how was work mainly affected?
table covwk_howaffect if covwk==1 & covwk_stop==0 [fw=round(wgtr_c)], stat(frequency) stat(sumw) stat(rawpercent) stat(percent) miss 

* how was work affected: riskier?
table covwk_risk if covwk==1 & covwk_stop==0 [fw=round(wgtr_c)], stat(frequency) stat(sumw) stat(rawpercent) stat(percent)

* how was work affected: harder?
table covwk_hard if covwk==1 & covwk_stop==0 [fw=round(wgtr_c)], stat(frequency) stat(sumw) stat(rawpercent) stat(percent)

* how was work affected: wfh?
table covwk_wfh if covwk==1 & covwk_stop==0 [fw=round(wgtr_c)], stat(frequency) stat(sumw) stat(rawpercent) stat(percent)

* create new labels 
gen covwk_overall = . 
replace covwk_overall = 0 if covwk==1 & covwk_affect==0 
replace covwk_overall = 1 if covwk==1 & covwk_affect==1 & covwk_stop==0
replace covwk_overall = 2 if covwk==1 & covwk_stop == 1 
label define overall 0 "Work not affected" 1 "Work affected" 2 "Work stopped", replace
label values covwk_overall overall 
tab covwk_overall

* detailed 
gen covwk_det = . 
replace covwk_det = covwk_event if covwk_event<. 
replace covwk_det = 4 if covwk_event==. & covwk_stop==1 // missing covwk_event goes to "other" 
replace covwk_det = 5 + covwk_howaffect if covwk_howaffect<7
replace covwk_det = 10 if covwk_howaffect==7 
replace covwk_det = 10 if covwk_howaffect==. & covwk_affect==1 & covwk_stop==0
replace covwk_det = 11 if covwk_affect==0 & covwk==1
tab covwk_det covwk_overall
label define detail 1 "Permanent job loss" 	/// 
					2 "Temporary layoff"  	///
					3 "Quit"				///
					4 "Other stop"			///
					5 "Retired"				///
					6 "Schedule change"		///
					7 "Work got riskier"	///
					8 "Work got harder"		///
					9 "Switched to remote"  ///
					10 "Other change to work" ///
					11 "No change" , replace 
label values covwk_det detail 
tab  covwk_det
		

************ THIS IS IT THIS IS THE GRAPH I WANT ***************
gen freq=1
bys covwk_det: egen covwk_det_tot=sum(wgtr_c) if covwk_det<.

* add broader category %s to label so it shows up in graph 
total wgtr_c if covwk==1 
local total = r(table)[1,1]
total wgtr_c if covwk==1 & covwk_overall==0
local share0 = round((r(table)[1,1]/`total')*100,1)
total wgtr_c if covwk==1 & covwk_overall==1
local share1 = round((r(table)[1,1]/`total')*100,1)
total wgtr_c if covwk==1 & covwk_overall==2
local share2 = round((r(table)[1,1]/`total')*100,1)
label define overall 0 "Work not affected (`share0'%)" ///
					 1 "Work affected (`share1'%)" ///
					 2 "Work stopped (`share2'%)", replace
label values covwk_overall overall 
graph hbar (percent) freq if covwk==1 [fw=round(wgtr_c)], ///
	over(covwk_det, sort(covwk_det_tot) desc label(labsize(*0.7)))  ///
	over(covwk_overall, label(labsize(*0.9))) nofill ///
	blabel(bar, position(outside) format(%9.1f)) ytitle("Percent") 
graph export "/Users/owen/Covid5/output/figs/hrs_covid_affect.pdf", replace


* -------------------------------------------------------------------------- * 
* ------------------------ 	collapse retired ------------------------------- * 
* -------------------------------------------------------------------------- * 
use hrs_monthly_temp, clear 
keep if date_death==.

sum iwdate if wave==14
scalar mindate = r(min)
sum iwdate if wave==15
scalar maxdate = r(max)

gen retired=lbrff==3
gen employed=lbrff==1 
gen n = dead==0 & wgtr>0 & wgtr<.
gen nw = n==1 & iwdate==mo 

bys wave: egen total = sum(nw)

collapse (mean) retired employed (rawsum) n (min) total  [fw=wgtr], by(mo)
tsset mo 
gen share = n/total

reg retired mo if mo>mindate & mo<`=tm(2020m3)' & share>0.9
predict p_retired

twoway tsline retired p_retired if inrange(mo,mindate,maxdate), ///
	xline(`=tm(2020m3)', lc(black) lp(dot)) /// 
	xlab(`=tm(2018m1)'(12)`=tm(2022m1)', format(%tmCY)) /// 
	xtitle("") /// 
	xsize(7.5) ///
	legend(order(1 "Retired share" 2 "Linear prediction")) 
graph export "/Users/owen/Covid5/output/figs/hrs_retired_share.pdf", replace



* -------------------------------------------------------------------------- * 
* ------------------------ 	hist of retire dates --------------------------- * 
* -------------------------------------------------------------------------- * 
use hrs_monthly_temp, clear 

* when retired 
sum iwdate if wave==13
scalar mindate = r(max)
sum iwdate if wave==15
scalar maxdate = r(max)

gen date_retired = mo if lbrff==3 & l.lbrff<3 & l.lbrff<. 
tab date_retired if emp13==1 & (inw15==1 | inw16==1)

twoway hist date_retired if emp13==1 & (inw14==1 & inw15==1 & inw16==1) ///
			& inrange(mo, mindate,maxdate) [fw=wgtr], color(black%50) lw(0.1) discrete ///
	  xla(,format(%tmCY)) xla(722 "March 2020", add angle(45)) ///
	  xline(`=tm(2020m3)', lc(black%50) lp(dash)) ///
	  xtitle("")  ///
	  plotregion(margin(zero) lwidth(none))
graph export  "/Users/owen/Covid5/output/figs/hrs_retire_date.pdf", replace




