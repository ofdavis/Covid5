clear all 
cd "/Users/owen/Covid5/Data/HRS/raw/sects/"

* -------------------------------------------------------------------------- * 
* ------------------------ 	descriptive	         --------------------------- * 
* -------------------------------------------------------------------------- * 
use hrs_monthly_temp, clear 

* redo dates for those who give whyleave=[covid] but date before covid 
* this could be for a couple reasons: date_empchg given but refers to earlier empchg (eg, emp->nilf pre covid, nilf->ret post covid)
* could also be imputed incorrectly by me as midpoint of dates where one is precovid 

* not retired in 2016
tab  lbrff_endw emp13 if tag_hw & wave==15 & inw13==1, col

* of those working in wave w and retired in wave w+t, reasons for leaving 
tab whyleave_any covid if empchg==1 & emp13==1 & inrange(wave,14,15) & lbrff==3 , miss col 
tab whyleave_any covid if empchg==1 & emp13==1 & inrange(wave,14,15) & lbrff_endw==3 [fw=wgtr], miss col nofreq

foreach var in ret_health ret_dothings ret_nowork ret_family { 
	tab `var' if tag_hw==1 & emp13==1 & wave==14 & lbrff_endw==3, 
}




* -------------------------------------------------------------------------- * 
* ------------------------ 	CPS-like regression  --------------------------- * 
* -------------------------------------------------------------------------- * 
* goal: among those employed in wave 13, retirement outcomes pre-post covid 
* ironically, need wide data for this! 
frame change default 
use hrs_monthly_temp, clear 
keep if mo==iwdate
drop if dead
gen retired=lbrff==3

* create 13-only vars to merge back 
frame copy default tmp13, replace
frame change tmp13 
keep hhidpn wave mo lbrff empchg mstatf hhid pn subhh wgtr gender race foreign educ nurshm cendiv urbrur hh_numch hh_numad hh_rent hh_hhres hh_child hh_achck hh_ahous hh_amort hh_atoth hh_atotf hh_icap btwjob sayret shlt hltc jhours slfemp samemp jcten jcoccc jcindc wgihr wgiwk fsize union jcpen age jhours_flag slfemp_flag jcten_flag jcoccc_flag jcindc_flag wgihr_flag wgiwk_flag fsize_flag union_flag jcpen_flag ft lachck lahous lamort latoth latotf licap any_adult any_child sp_lbrff sp_dead sp_nurshm sp_shlt sp_age sp_ft covid retired

reshape wide mo lbrff empchg mstatf hhid pn subhh wgtr gender race foreign educ nurshm cendiv urbrur hh_numch hh_numad hh_rent hh_hhres hh_child hh_achck hh_ahous hh_amort hh_atoth hh_atotf hh_icap btwjob sayret shlt hltc jhours slfemp samemp jcten jcoccc jcindc wgihr wgiwk fsize union jcpen age jhours_flag slfemp_flag jcten_flag jcoccc_flag jcindc_flag wgihr_flag wgiwk_flag fsize_flag union_flag jcpen_flag ft lachck lahous lamort latoth latotf licap any_adult any_child sp_lbrff sp_dead sp_nurshm sp_shlt sp_age sp_ft covid retired, i(hhidpn) j(wave)

keep hhidpn *13

tempfile tmp13 
save "`tmp13'"
frame change default 
merge m:1 hhidpn using "`tmp13'"


* test regression like cps one ( I don't love this ... selection issues etc )

reghdfe retired covid##(c.age##c.age i.gender i.race i.mstatf i.foreign i.urbrur i.educ i.hh_rent i.shlt i.jcoccc13 i.ft13 i.slfemp13 c.jcten13 ///
		c.hh_numad c.hh_numch c.lahous c.latotf) [pw=wgtr] if lbrff13==1 & inw13==1 & (inw14==1 | inw15==1 | inw16==1) & inrange(wave,14,16), ///
	absorb(mo cendiv) cluster(hhidpn )

	
mat b = e(b)
mat list b
twoway function y=b[1,"age"]*x + b[1,"c.age#c.age"]*x^2 					, range(50 80) /// 
	|| function y=b[1,"1.covid#c.age"]*x + b[1,"1.covid#c.age#c.age"]*x^2 	, range(50 80) //
	
table wave retired if emp13==1 & inw13==1 & inw14==1 & inw15==1 & inw16==1 & inrange(wave,14,16)



* -------------------------------------------------------------------------- * 
* ------------------------ 	regression with covid -------------------------- * 
* -------------------------------------------------------------------------- * 
* goal: among those employed in wave 13, retirement outcomes pre-post covid 
* ironically, need wide data for this! 
frame change default 
use hrs_monthly_temp, clear 
keep if mo==iwdate
drop if dead
gen retired=lbrff==3

* create 13-only vars to merge back 
keep hhidpn wave incovid mo lbrff empchg mstatf hhid pn wgtr gender race foreign educ nurshm cendiv urbrur hh_numch hh_numad hh_rent hh_hhres hh_child hh_achck hh_ahous hh_amort hh_atoth hh_atotf hh_icap btwjob sayret shlt hltc jhours slfemp samemp jcten jcoccc jcindc wgihr wgiwk fsize union jcpen age jhours_flag slfemp_flag jcten_flag jcoccc_flag jcindc_flag wgihr_flag wgiwk_flag fsize_flag union_flag jcpen_flag ft lachck lahous lamort latoth latotf licap any_adult any_child sp_lbrff sp_dead sp_nurshm sp_shlt sp_age sp_ft covid retired covid_any wgtr_c covwk covwk_affect covwk_stop covwk_event covwk_findnew covwk_howaffect covwk_risk covwk_hard covwk_wfh covwk_ownbiz covwk_ownaffect covwk_ownclose covwk_ownpermclose covid_concern died*

reshape wide mo lbrff empchg mstatf wgtr nurshm cendiv urbrur hh_numch hh_numad hh_rent hh_hhres hh_child hh_achck hh_ahous hh_amort hh_atoth hh_atotf hh_icap btwjob sayret shlt hltc jhours slfemp samemp jcten jcoccc jcindc wgihr wgiwk fsize union jcpen age jhours_flag slfemp_flag jcten_flag jcoccc_flag jcindc_flag wgihr_flag wgiwk_flag fsize_flag union_flag jcpen_flag ft lachck lahous lamort latoth latotf licap any_adult any_child sp_lbrff sp_dead sp_nurshm sp_shlt sp_age sp_ft covid retired covid_any wgtr_c covwk covwk_affect covwk_stop covwk_event covwk_findnew covwk_howaffect covwk_risk covwk_hard covwk_wfh covwk_ownbiz covwk_ownaffect covwk_ownclose covwk_ownpermclose covid_concern, i(hhidpn) j(wave)

foreach var in covwk covwk_affect covwk_stop covwk_event covwk_findnew covwk_howaffect covwk_risk covwk_hard covwk_wfh covwk_ownbiz covwk_ownaffect covwk_ownclose covwk_ownpermclose covid_concern { 
	drop `var'13 `var'14 `var'16
	rename `var'15 `var'
	replace `var'=0 if `var'==.
}
gen covwk_lose = covwk_event==1
gen covwk_layoff = covwk_event==2
gen covwk_quit = covwk_event==3
gen covwk_retire = covwk_event==5

replace sp_lbrff13=0 if sp_lbrff13==.
replace sp_lbrff14=0 if sp_lbrff14==.
replace sp_age13=0 if sp_age13==.
replace sp_age14=0 if sp_age14==.

* among those working in 2016/2018 and in covid sample, predict retirement in 2022 
gen samp13 = lbrff13==1 & incovid==1 & wgtr_c16<. & wgtr_c16>0 & covwk==1
gen samp14 = lbrff14==1 & incovid==1 & wgtr_c16<. & wgtr_c16>0 & covwk==1


local st 14
sum retired16 if samp`st'==1 [fw=round(wgtr_c`st')] 

local st 14
reghdfe retired16 c.age`st'##c.age`st' ///
	covwk_lose  covwk_wfh covwk_hard covid_concern  ///    covwk_risk  gender  
	[pw=wgtr_c`st'] if samp`st'==1, absorb()
	
local st 14
reghdfe retired16 c.age`st'##c.age`st' i.mstatf`st' i.mstatf`st'#c.sp_age`st' i.sp_lbrff`st' i.shlt`st' ///
	hh_hhres`st' hh_ahous`st' hh_atotf`st' hh_rent`st' ///
	slfemp`st'  jhours`st' wgiwk`st' jcten`st'  /// 
	covwk_lose covwk_wfh covwk_hard covid_concern  ///  c.  covwk_risk    
		[pw=wgtr_c`st'] if samp`st'==1, ///
		absorb(i.cendiv`st' i.educ gender foreign i.urbrur`st' i.race i.jcoccc`st' i.jcindc`st' i.mo16)
	
* lasso then use lasso-selected covars 
local st 14
lasso linear retired16 c.age`st'##c.age`st' i.mstatf`st' i.mstatf`st'#c.sp_age`st' i.sp_lbrff`st' i.shlt`st' ///
	slfemp`st' jhours`st' jcten`st'  wgiwk`st' ///
	hh_hhres`st' hh_ahous`st' hh_atotf`st' hh_rent`st' ///
	covwk_lose covwk_wfh covwk_hard covid_concern ///
	i.cendiv`st' i.educ gender foreign i.urbrur`st' i.race  i.jcoccc`st' i.jcindc`st' i.mo16 ///
	 if samp`st'==1, selection(cv, alllambdas)
	 
lassocoef
local st 14
local vars = e(allvars_sel)
reghdfe retired16 `vars' ///
	 covwk_lose  covwk_wfh covwk_hard covid_concern ///    covwk_risk    
	[pw=wgtr_c`st'] if samp`st'==1, 

* logit 	 
local st 14
logit retired16 c.age`st'##c.age`st' i.mstatf`st' i.mstatf`st'#c.sp_age`st' i.sp_lbrff`st' i.shlt`st' ///
	slfemp`st' jhours`st' jcten`st'  wgiwk`st' ///
	hh_hhres`st' hh_ahous`st' hh_atotf`st' hh_rent`st' ///
	i.cendiv`st' i.educ gender foreign i.urbrur`st' i.race  i.jcoccc`st' i.jcindc`st' i.mo16 ///
	covwk_lose  covwk_wfh covwk_hard covid_concern  ///    covwk_risk  
	[pw=wgtr_c`st'] if samp`st'==1	 
	
* ------------ predictors of covid outcomes --------------- * 
foreach var in covwk_stop covwk_lose covwk_wfh covwk_hard covid_concern { 
	local st 14
	reghdfe `var' c.age`st'##c.age`st' gender i.educ i.race foreign ///
	i.mstatf`st' i.sp_lbrff`st' i.shlt`st' i.cendiv`st'  i.urbrur`st'  ///
	slfemp`st' jhours`st' jcten`st'  wgiwk`st' i.jcoccc`st' i.jcindc`st' ///
	hh_hhres`st' hh_ahous`st' hh_atotf`st' hh_rent`st' ///  
	[pw=wgtr_c`st'] if samp`st'==1, absorb()
	est sto `var'
}

esttab covwk_stop covwk_lose covwk_wfh covwk_hard covid_concern, nogap nobase label

tab covwk_event if incovid==1 & covwk==1 [fw=round(wgtr_c15)]

gen nilf16=inrange(lbrff16,3,5)
	
* -------------------------------------------------------------------------- * 
* ------------------------ create survival data  --------------------------- * 
* -------------------------------------------------------------------------- * 
use hrs_monthly_temp, clear 

cap program drop survset 
program define survset
	syntax, Vars(string) [test Nocoll]
	di "`nocoll'.............."
	cap drop spell 
	local add "" 
	foreach v in `vars' { 
		cap drop d_`v'
		gen d_`v' = l.`v'!=`v'
		local add = "`add' " + "+ d_`v' "
	}
	di "`add'"
	gen spell = 1
	replace spell = spell[_n-1] `add' if l.spell<.
	
	* length of spell 
	bys hhidpn spell (mo): gen length = _N
	xtset hhidpn mo 
		
	* indicate types of endings: retired, dead, missing 
	gen end_dead_ = 0 
	replace end_dead_ = 1 if dead==0 & f.dead==1

	gen end_retire_ = 0 
	replace end_retire_ = lbrff!=3 & f.lbrff==3 & dead==0
	
	gen end_unretire_ = 0 
	replace end_unretire_ = lbrff==3 & f.lbrff!=3 & dead==0 & f.dead==0

	gen end_miss_ = 0 
	bys hhidpn (mo): gen last = _n==_N
	replace end_miss_ = last==1 & dead==0 & wave<16

	foreach var in dead retire unretire miss {
		bys hhidpn spell (mo): egen end_`var' = max(end_`var'_)
	}

	* testing 
	if "`test'"=="test" {
		tab end_dead end_retire
		tab end_dead end_miss 
		tab end_retire end_miss 
	}	
	
	if "`nocoll'"!="nocoll" {
		bys hhidpn spell: keep if _n==1
	}
	else {
		di "not collapsed"
	}

	drop end_miss_ end_retire_ end_unretire_ end_dead_ d_* 
end

survset, v(lbrff covid) 


* ------------------------- surv1: only employment changes, no covid --------------------* 
frame2 surv1, replace 
use hrs_monthly_temp, clear 

gen widowed = mstat==7 & inrange(l.mstat,1,3)
gen newpart = inrange(mstat,1,3) & inrange(l.mstat,4,8)

survset, vars(lbrff mstat) //nocoll test

drop if lbrff==3

stset length, failure(end_retire) id(hhidpn) 

sts graph, by(mstat)

stcox c.age##c.age##c.age gender i.race i.educ hh_numch hh_numad i.lbrff i.shlt i.mstatf widowed newpart i.cendiv i.urbrur foreign


* ------------------------- surv1_c: only employment changes, with covid --------------------* 
frame2 surv1_c, replace 
use hrs_monthly_temp, clear 

survset, vars(lbrff covid mstat) 


stset length, failure(end_retire) id(hhidpn) 
keep if lbrff!=3

sts graph, by(covid)


stcox i.covid##(c.age##c.age gender i.race i.educ c.hh_numch c.hh_numad i.lbrff i.shlt i.mstatf i.cendiv##i.urbrur foreign)

stcox i.covid##(c.age##c.age gender i.race i.educ c.hh_numch c.hh_numad i.shlt i.mstatf i.urbrur foreign c.ft i.slfemp c.lahous c.latotf c.jcten c.wgiwk) if lbrff==1

stcox i.covid##(c.age##c.age), 
stcrreg i.covid##(c.age##c.age), compete(end_dead=1) 
stcrreg i.covid##(c.age##c.age gender i.race i.educ c.hh_numch c.hh_numad i.shlt i.mstatf i.urbrur foreign c.ft i.slfemp c.lahous c.latotf c.jcten c.wgiwk) if lbrff==1, compete(end_dead==1)


* ------------------------- surv1_c: emp changes, with covid -- outcome death  --------------------* 
frame2 surv1_c_d, replace 
use hrs_monthly_temp, clear 

survset, vars(lbrff covid mstat) 


stset length, failure(end_dead) id(hhidpn) 

sts graph, by(covid)


stcox i.covid##(c.age##c.age gender i.race i.educ c.hh_numch c.hh_numad i.lbrff i.shlt i.mstatf i.cendiv##i.urbrur foreign)

stcox i.covid##(c.age##c.age gender i.race i.educ c.hh_numch c.hh_numad i.shlt i.mstatf i.urbrur foreign c.lahous c.latotf c.jcten c.wgiwk) if lbrff==1


*--------------------------------- test 2: died regs  ------------------------------------
frame2 test, replace 
use hrs_monthly_temp, clear 

gen retired=lbrff==3
gen dum = 1 

gen died = f.dead==1 
gen f_covid = mo>=`=tm(2020m2)'
gen retirement = lbrff!=3 & f.lbrff==3

* died reg 
reghdfe died f_covid##(c.age##c.age##c.age i.educ i.shlt i.mstatf i.lbrff i.cendiv c.hh_numad c.hh_numch i.nurshm hh_rent ) if f.hhidpn<. [pw=wgtr], vce(cluster hhidpn)

reghdfe died f_covid##(c.age##c.age##c.age i.educ i.shlt i.mstatf i.lbrff i.cendiv i.urbru	 c.hh_hhres hh_rent ) if f.hhidpn<. [pw=wgtr], vce(cluster hhidpn)



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
twoway tsline retired p_retired if inrange(mo,mindate,maxdate), xline(`=tm(2020m3)')

reg employed mo if mo>mindate & mo<`=tm(2020m3)' & share>0.9
predict p_employed
twoway tsline employed p_employed if inrange(mo,mindate,maxdate), xline(`=tm(2020m3)')



* -------------------------------------------------------------------------- * 
* ------------------------ 	collapse transitions --------------------------- * 
* -------------------------------------------------------------------------- * 
* this is not helpful -- too noisy 
use hrs_monthly_temp, clear 
keep if date_death==.

sum iwdate if wave==14
scalar mindate = r(min)
sum iwdate if wave==15
scalar maxdate = r(max)

gen lr = l.lbrff==1 & l.lbrff<. & lbrff==3 
gen rl = l.lbrff==3 & lbrff==1 & lbrff<.
gen n=1

collapse (sum) lr rl n if dead==0 [fw=wgtr], by(mo)
tsset mo 

gen lr_pct = lr/n 
gen rl_pct = rl/n 

tsline lr_pct rl_pct  if inrange(mo,mindate,maxdate), xline(`=tm(2020m3)')






