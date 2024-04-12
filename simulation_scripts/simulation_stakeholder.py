import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt


# Load environment variables from .env file
load_dotenv()

key = os.getenv('OPENAI_API_KEY')
if key is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set \
                     or .env file is missing.")

client = OpenAI(
    api_key=key
)

def generate_user_prompt():
    prompt = """
        International Staff of Health for All (HfA) Detained by Tribesmen in Iguwafe


        The international health organization, Health for All (HfA), based in Geneva, offers medical aid to those affected by conflicts globally. Due to increased ethnic clashes in Country Beta in early 2020, HfA deployed a full surgical unit at Iguwafe's rural district hospital, specializing in war surgery for civilians injured in these hostilities. 

        By 2023, as local tensions subsided due to tribal leaders' mediation, the demand for surgical care shifted towards other sources of injuries like road accidents, domestic violence, obstetrics and burns as well as reconstructive surgery for many trauma patients left with debilitating scars. While the surgical needs are real, many patients fall outside the original mission of HfA. No other surgical care providers operate in the district due to resource constraints and social unrest.

        Simultaneously, HfA experienced a pressing need to redirect medical resources to conflict-stricken Ukraine under the pressure from its main donors, Germany and Switzerland. As a result, a decision was made in late September 2023 at HfA's headquarters to transfer the surgical team from Beta to Ukraine in the best delay. This decision, made with limited local consultation, meant closing trauma-related operations in Beta. The sudden transfer of HfA's Country Director to Kiev in September sent shockwaves through HfA in Beta leaving the local staff with minimal guidance on the upcoming closure.

        Rumors of HfA's departure spread rapidly, causing concerns among the local hospital staff and community reliant on HfA's presence. An unexpected protest occurred outside HfA's Iguwafe office in late September, where demonstrators demanded answers from HfA's leadership. Matters escalated when a junior HfA member confirmed the Country Director's departure and revealed the irreversible character of the closure of surgery operations at the end of the week. Outraged by this information, local armed tribesmen moved promptly to impose a house arrest on nine HfA international staff members, threatening them with violence if they attempted to leave their residence.

        Tribal leaders undertook to mediate the tension between the local population and HfA. They conveyed the population’s discontent with HfA's abrupt decision and emphasized the critical role HfA played in the district's healthcare. They expressed worries about rising measles cases in the region and the effects of the potential closure of the hospital without HfA's support. Additionally, they demanded that HfA provide benefits to the hospital's guards, who had faced immense risks during the conflict. Some of these guards had tragically lost their lives or suffered long-term disabilities.

        Due to the enforced house arrest of the HfA international staff, the district hospital's operations have been disrupted ever since, leaving many patients without essential care. The lack of surgical care has caused significant distress among the patients' families.

        While the Government in the capital expressed its discontent about the house arrest, the government has remained distant, viewing this conflict as a private labor dispute. The local army and police have refrained from intervening considering their limited influence on the tribesmen without tribal chiefs' endorsement.

        In an attempt to resolve the situation, HfA representatives from Geneva plan to meet with the Tribal Leaders. HfA's primary focus now is to strategize negotiations for the safe release of its staff detained by the tribesmen, ensuring their security and well-being.



        Additional information see the following:


        1. Source: Report of the World Health Organization Regional Office

        Country Beta has faced protracted community violence, political instability, and natural disasters like droughts and floods, which have significantly impacted its healthcare infrastructure and services. As a result of these combined factors, many healthcare facilities in the country are either non-functional or partially operational. This has led to limited access to basic healthcare services for a vast majority of the population. Surgical care is mostly absent from the rural areas of Beta with the exception of the trauma care offered by international NGOs. Furthermore, Beta has a high burden of infectious diseases such as tuberculosis, cholera, and measles. The immunization coverage remains low, leading to periodic outbreaks of preventable diseases like polio and measles, particularly in rural areas.

        Inadequate access to clean water and sanitation facilities has contributed to recurrent outbreaks. The country has one of the highest maternal mortality rates in the world mostly due to the inadequate access to maternal health services, skilled birth attendance, obstetric surgery and family planning.

        International organizations and NGOs play a crucial role in providing health services, especially in remote and conflict-affected areas. However, access can be hampered by security concerns, particularly with the increasing reliance on tribesmen to ensure law and order in some of the districts.


        2. Source: Meeting with the Minister of Health of Beta

        The Health Minister warmly greeted the HfA team arriving from Geneva. The Beta government conveys its profound appreciation, especially given the challenges faced by HfA. The Minister stated, “Our nation genuinely relies on international support from esteemed NGOs like HfA to deliver healthcare to our people.” The Geneva delegation highlighted their immediate goal of ensuring the safe release of their staff held in Iguwafe, stating that, “Such incidents shouldn't occur. Our primary concern remains our team's safety.” They further elaborated on the mounting global pressure to shift medical aid to Ukraine, necessitating discussions with the Beta government about transitioning their health activities. According to HfA, with the decreasing tension in Beta, it’s deemed appropriate for humanitarian organizations to redirect their focus to areas with pressing needs, such as war-affected Ukraine. The Minister acknowledged the predicament of Health for All, adding, “It's disheartening to witness global humanitarian organizations depart, especially towards European countries that already have substantial resources. Their departure will undeniably impact our citizens deeply. I cannot hide my fear that low-resource government like ours will not be able to fill this void.” When queried about the possibility of governmental action for the staff's release in Iguwafe, she responded, “The matter largely rests with tribal leaders. Any governmental interference might escalate tensions. Continuous dialogue with tribal heads seems the most prudent course.”


        3. Source: Meeting with the UN Humanitarian Coordinator in Beta

        The UN Humanitarian Coordinator expressed profound gratitude for "Health for All's" endeavors in Beta. The UN is deeply concerned about the worsening security conditions in the country's rural regions, with the incident involving Health for All's staff being particularly alarming. Predominantly, security in these areas is managed by tribal chiefs, with minimal intervention from government or military forces.

        The UN was taken aback by Health for All's sudden decision to leave Beta without extensive consultations with the local government or tribal leaders. This move is uncharacteristic of the organization's reputed standards. The HfA team from Geneva clarified that a swift operational shift was required by increasing demand for surgical care in Ukraine, which led to the prompt relocation of the country director to Kiev. The lack of leadership in Beta caused a communication gap regarding the organization's changing priorities. This oversight was regrettable.

        The HfA team affirmed that given the circumstances, Health for All is now keen on ensuring open dialogue with communities and tribal leaders, especially concerning the release of their staff. The organization stands firm on its redeployment to Ukraine due to depleted funds for Beta operations. Major donors like Germany and Switzerland have mandated the transition of all medical resources and staff to Ukraine. The objective now is to transition the Iguwafe hospital's control to local health officials, with the financial assistance from developmental agencies. We should note that Health for All's surgical team in Iguwafe, primarily trained for war-related injuries, mostly cater to road accident victims.

        The UN Coordinator pointed out that while global conflicts indeed result in wartime casualties, Beta's dilapidated health infrastructure makes it resemble a perpetual conflict zone. It's distressing that global aid organizations are pivoting towards richer nations despite the continued humanitarian needs in places like Beta.

        The UN Security Coordinator, also in attendance, urged the Geneva team to foster meaningful conversations with tribal leaders. It is, in his view, the only way to bring about the safe release of the staff. However, caution was advised concerning local security personnel, many of whom had affiliations with violent factions during past conflicts. These guards can be unpredictable and potentially dangerous, as can some tribesmen. Ensuring the safety of Health for All's staff and residences is paramount. Recent intel from partners in Iguwafe suggests potential demands for ransom by tribesmen for the release of HfA staff. The UN Security Coordinator suggested hiring expert negotiators to handle the hostage situation, mentioning that the UN has reliable contacts who can ensure minimal ransom payouts.

        The Health for All representatives expressed gratitude to the UN Security Coordinator for his insights and recommendation. However, they clarified that their organization adheres to a strict no-ransom policy in hostage situations. If they begin paying ransoms in one region, they might face similar demands elsewhere, compromising their principles. Meanwhile, no ransom have been demanded so HfA will insist to deal with this situation as a labor dispute rather than a hostage situation. The UN Security Coordinator wished the Geneva team well and reiterated his offer of providing seasoned negotiators.


        4. Source: Meeting with the Country Director of Food Without Border in Beta

        Food Without Borders (FWB) has been present in the country for over twenty years. Its Country Director, a seasoned professional with a wide network of connections, expressed shock over the detention of Health for All's staff in Iguwafe. He's deeply concerned that such incidents might set a dangerous precedent and endanger the security of many humanitarian workers in the country. The team from Health for All confirmed that their primary concern is the safe release of their staff as HfA will be leaving Beta soon. The team explains that due to funding constraints, HfA is compelled to shift most of its resources to Ukraine, discontinuing operations in Beta and many other poorly funded operations.

        The FWB Country Director very much understand this situation as his organization faces similar financial challenges. Despite these limitations, he believes it's implausible to exit Beta  under the prevailing circumstances. In his view, while the ongoing humanitarian crisis in Beta remains severe, the global media's portrayal of the war in Ukraine makes the crisis in Beta less enticing for donors. Instead, government donors are inclined to support military and humanitarian operations in countries like Ukraine, much more visible in the press and government circles. This scenario underscores the politicization of humanitarian aid, leaving relief workers like us to navigate these complexities in Beta. 

        For FWB, the top priority is the safe release of the detained staff in Iguwafe. The Country Director suggested that for Health for All to see their staff safely returned, they will need to strike a compromise with the tribal leaders and the community, such as maintaining some medical activities at the hospital. He highlighted the growing unrest in places like Iguwafe. With the diminishing employment opportunities for local guards and tribesmen formerly associated with different factions, some are exploring alternative income sources. Sustainable safety for humanitarian staff in rural areas can only be achieved with the backing of tribal leaders and the local population. This perspective should be taken into account as Health for All evaluates their next steps regarding the detained staff.

        FWB pledges its full support to ensure the staff's release, including leveraging its rapport with tribal leaders. Having significantly aided the community during food security crises, especially among children, FWB's commitment remains unwavering.


        5. Source: Minutes of Meeting with the German Ambassador

        Date: October 10, 2023  
        Location: German Embassy  
        Attendees: German Ambassador, HfA Team from Geneva

        a. Welcome and Introduction  
        German ambassador received the HfA team at the embassy.

        b. Discussion on HfA's Operations and Security Incident  
        German ambassador inquired about HfA’s operational status in the country.
        She also asked about the recent security incident in Iguwafe.
        HfA team reported that the staff has been detained in their residence for several days, unable to leave until an agreement is reached between the organization and tribal leaders concerning health service maintenance in the district.
        HfA team mentioned challenges due to tribesmen guarding the residence and their vested interest in the deal.

        c. Government's Policy on Humanitarian Aid  
        German ambassador expressed concerns regarding the diversion of humanitarian aid to maintain development program. She sees the current security incident as an attempt to use extortion of HfA, and by extension the German government, to ensure the continuity of the dependency of Beta on international aid.
        She emphasized that humanitarian aid should be allocated to conflict situations and not diverted to developmental issues. There are other programs of aid for development issue for which the government of Beta must be accountable.
        Highlighted that countries should prioritize development themselves and not rely on humanitarian aid for such purposes.
        Mentioned other organizations and frameworks are better suited for development projects.

        d. Focus on HfA's war surgery specialization 
        The ambassador stressed the need for HfA to shift its operations to Ukraine due to the ongoing armed conflict there.
        Pointed out that HfA's surgical teams specialize in war surgery, making it more sensible to operate in conflict zones like Ukraine rather than places like Beta.

        e. Safety Concerns of HfA Staff  
        German ambassador expressed her concerns regarding the safety of the detained staff.
        Voiced confidence in HfA’s professionalism, believing the organization would find swift solutions for the safe release of the staff.

        f. Possible Compromise and Support  
        The HfA team asked the ambassador about potential compromises or solutions on the maintenance of health programs in Iguwafe for the coming months to secure the staff's release.
        The ambassador clarified that such decisions are beyond her authority and should be discussed with officials in Berlin.
        Her current directive is to support the safe release of the staff and assist in redirecting humanitarian organizations to areas facing conflict crises.

        6. Source: Minutes of Meeting with the Swiss Ambassador

        Date: October 10, 2023  
        Location: Swiss Embassy  
        Attendees: Swiss Ambassador, Health for All (HfA) Team

        a. Welcome and Introduction  
        The team from Health for All met with the Swiss Ambassador in Beta.

        b. Report Presentation by the Geneva Team  
        The Swiss Ambassador attentively listened to the report from the HfA Team.
        He concurred with the German Ambassador's viewpoint about HfA's need to relocate its operations to Ukraine, considering the complex situation in Beta and its development needs

        c. Discussion on the Detained Staff's Nationality  
        The Ambassador inquired about the nationality of the staff detained in Iguwafe, presuming they might be Swiss nationals.
        The HfA Team confirmed that all members of the surgical team in Iguwafe are Swiss nationals.
        The Swiss government is prepared to offer additional support to secure their release, such as hostage negotiation experts from the Federal Police in Bern.
        The Health for All team thanks the Ambassador for this offer but decline to have Federal Police involved in a situation that no one wants to call a hostage crisis. The Ambassador understands the situation.

        d. Debate on Decreasing Activities vs. Terminating Services  
        The HfA Team proposed a potential flexibility from the Swiss donor, allowing HfA to scale down its activities at the hospital rather than threatening an abrupt cessation of services which prompted the current crisis.
        The Swiss Ambassador expressed concerns about this strategy merely postponing the inevitable withdrawal of HfA from servicing developmental needs in Beta. He advocated for a development NGO to take over the hospital's health services, recognizing the difficulty to find such agency due to the region's instability and the lack of experience of these development organization to work in tense tribal areas.

        e. Discussion on the Paradoxical Situation in Beta  
        Both parties acknowledged the irony that despite decreased conflict thanks to the tribal leaders mediation, no development agencies are available to function in such an unstable environment. They wait for the government to regain control over the area which would prompt further violence with the tribesmen. Essentially, no one to endorse the role of tribal leaders by providing development aid in these circumstances.

        f. Safety and Security Concerns  
        The HfA Team voiced serious worries about the safety of staff, with fears of possible ransom demands from tribesmen and unpredictable actions from local guards if a solution is not found promptly.
        They highlighted the necessity for donor flexibility to ensure smooth transitions in countries like Beta.

        g. Swiss Ambassador's Recommendations and Commitment  
        The Ambassador cautioned against paying any ransoms to local mafias.
        He endorsed collaboration with the UN coordinator, who possesses significant experience in such situations.
        The Ambassador is committed to communicate with the Swiss headquarters in Bern about allocating special funds to sustain minimal health services at the hospital, such as support to deal with the emerging measle problem among children, aiming for a compromise to facilitate the release of international staff.
        However, he will advise the repatriation of all Swiss nationals to Switzerland and suggests that HfA source alternative surgeons for the hospital operations.


        7. Source: Meeting with Child Protection International (CPI)

        Child Protection International (CPI) is a renowned human rights organization focusing on the welfare and well-being of children affected by armed conflicts. Health for All met with representatives of CPI in Beta upon the latter's request. The primary objective of the meeting was to discuss the ongoing medical needs in the Iguwafe region and the redeployment concerns of the Health for All surgical team.

        CPI expressed profound concerns over the potential redeployment of the health team, especially given the dire medical needs in Beta. The Health Minister, closely affiliated with CPI due to familial connections, requested CPI's intervention to persuade Health for All to continue its operations in Beta. CPI underscored the urgent requirements for fighting the measle epidemy as well as for post-operative care in the region. CPI highlighted the invaluable contributions of the Health for All surgical team that performed emergency surgeries on children victims violent incidents. However, many of these children now require reconstructive plastic surgery to address the scars and subsequent physical, social and mental implications of the violence which continue to affect them. Emphasizing the dire consequences faced by these children - many of whom can't attend school or even leave their homes due to visible scars and its associated social stigmas- the CPI representative made a heartfelt plea for Health for All to find resources and offer the necessary plastic surgeries.

        The HfA Team clarified that while they desire to continue providing assistance, the specialized nature and high costs associated with plastic surgery pose significant challenges. Health for All acknowledged the grave situation and mentioned exploring collaborations with organizations like Swisscross, which might have the expertise and resources for plastic surgery. Ensuring the safe release of their detained staff in Iguwafe remains a pressing concern for Health for All. HfA hopes for CPI's support in engaging with tribal leaders to facilitate this. CPI's representative assured Health for All of their complete backing in efforts to secure the release of the detained staff and expressed hope for continued medical support in the region.


        8. Source: Local press

        Local Newspapers Summary: Health for All's Controversial Decision Sparks Outrage in Iguwafe

        The decision of "Health for All" (HfA) to pull out from Iguwafe without comprehensive consultation with local stakeholders and plans for alternative health solutions has faced strong condemnation from the local media. Reports highlighted the significant demonstration outside HfA's offices, reflecting the community's anger, especially upon discovering the abrupt departure of the country director without engaging local authorities.

        The media praised the pivotal role of local guards during previous crises, highlighting their contributions in ensuring the safe delivery of essential healthcare services. Now, these guards, once hailed as heroes, are left uncompensated. Similarly, local hospital employees face impending unemployment.

        The local media interprets HfA's decision as an attempt to exert foreign influence over the region, terming it "humanitarian colonialism" and urging for its reversal. While the detainment of HfA staff has been marginally covered, the halt of hospital activities amidst this upheaval was underscored. Concerns rise as families of patients demand answers from local authorities and tribal leaders regarding the immediate resumption of healthcare provisions. Some people believe that tribesmen are using this situation to extort money from HfA by preventing the international staff to work at the hospital.


        9. Source: Meeting with the local head of police in Iguwafe

        Report: Meeting with Local Police Chief on HfA Staff Situation

        The Health for All (HfA) delegation recently met with the local police chief to discuss the ongoing staff situation. During the meeting, the chief expressed his discontent, firmly placing the responsibility for the situation on HfA.

        The police chief emphasized the importance of HfA modifying its policy and continuing minimal health services at the local hospital for the time being. While the police are committed to facilitating the safe release of the detained staff, the chief noted that a resolution seems improbable amidst the current labor disagreements involving local authorities, hospital personnel, and tribal leaders.

        Although the staff is deemed safe within their residence and have even been permitted to venture out for essential supplies, the chief warned of a potentially deteriorating situation if HfA fails to negotiate swiftly with the tribal leaders. He was clear in stating the police will not intervene, fearing potential violent clashes with the tribesmen.

        The chief's recommendation to the visiting HfA team from Geneva is to engage in a constructive dialogue with the tribal leaders. Once an agreement on the health services is reached, the police will step in to restore peace.


        10. Source: Meeting with the local head of the District Administration

        Report: Meeting with Local Administration Head in Iguwafe Regarding HfA Situation

        The Health for All (HfA) delegation from Geneva recently held discussions with the head of the local administration in Iguwafe. Interestingly, the local administrator is an alumnus of the University of Geneva's School for Humanitarian Services and is therefore well-acquainted with the intricacies of international humanitarian efforts.

        The administrator acknowledges the mounting pressures on HfA to relocate its resources to more pressing regions like Ukraine. He emphasizes that while war-affected countries like Ukraine undoubtedly require humanitarian support, there is a dire need for the Beta national government to direct its own funds to its rural sectors. 

        In his assessment, the continuous presence of organizations like HfA has unintentionally enabled the national government to channel resources primarily to urban centers, consequently neglecting healthcare necessities in rural regions like Iguwafe. He expressed his support for HfA's proposed transition from the local hospital and conveyed his readiness to collaborate with suitable partners to facilitate this change smoothly.

        The administrator is keen to liaise with tribal leaders to establish a government-led framework to enforce national health policies in the area. While he values the influential role of tribal leaders in halting past community conflicts, he asserts that such leaders should refrain from involving themselves in healthcare-related issues and labor disputes within the hospital setting.

        In conclusion, the local head advises HfA to strategize the safe return of its staff to the hospital. He advocates for discussions between HfA and the national government to transition the delivery of health services to local organizations, ensuring that international NGOs aren't ensnared in domestic power struggle between the public sector and local stakeholders.


        11. Source: Meeting with representatives of the Patient Association of Iguwafe

        Report: Meeting with Patient Association Representatives in Iguwafe

        Upon arrival in Iguwafe, the Health for All (HfA) delegation was approached by representatives from the Patient Association, the same group that led the recent demonstration outside HfA's local office.

        The Patient Association began by expressing gratitude to HfA for visiting Iguwafe, seeing it as a sign of the organization's acknowledgment of its crucial role in healthcare delivery in the region. The Association is eager to collaborate with HfA to restore hospital services.

        They presented HfA with a list of patients requiring urgent surgical intervention, highlighting two individuals injured in a recent road accident who are currently without medical attention at their homes. The representatives also showed photos of children bearing scars and deformities from past inadequate treatments, emphasizing HfA's responsibility in ensuring these children receive necessary care to lead normal lives, including returning to school.

        Furthermore, the Association alerted HfA to a growing number of measles cases in the district, causing considerable alarm among local families. They urge HfA to spearhead vaccination campaigns, given the lack of substantial assistance from the national government on this front.

        In a concerning revelation, the representatives mentioned the financial struggles of the district's residents. With the ongoing issues at the local hospital, many are left with no choice but to seek expensive healthcare services in the capital. The Association fears that in the absence of funds, locals might turn to tribal affiliations to pressure organizations, including HfA, for financial resources.

        The crux of the discussion underscored the urgency of reinstating healthcare services at the local hospital. The Association firmly believes that immediate action will prevent further strain on relationships with humanitarian entities.

        The HfA delegation from Geneva listened intently to the concerns and requests presented by the Patient Association, inviting their support in ensuring the release of the staff and their return to their duty at the hospital.


        12. Source: Meeting with the head surgeon of HfA in Iguwafe, accompanied by two armed tribesmen.

        Upon concluding the meeting with the Patient Association representatives, the HfA delegation arrived at the HfA office. Unexpectedly, two armed individuals, presumably tribal guards, escorted the Head Surgeon of HfA in Iguwafe. They left the surgeon with the delegation for few minutes, issuing an explicit warning against any attempts to keep him from returning to his residence. The tribesmen guarded the door of the office.

        Once alone, the Head Surgeon described the current situation at the residence. Although telecommunications, including phones and the internet, were cut off, they have essential resources like food, water, and electricity. Armed guards monitor their every move, confining them to the premises with minimal outside contact. The team can send out one member daily for necessities. Overall, they face ennui due to their limited activities.

        He articulated the team's growing frustration with HfA's mismanagement of its relationship with local tribal leaders and the lack of clarity about the organization's plans. This mishandling has caused a significant breach of trust between HfA and the local community. Some team members, wary of the volatile security situation, are eager to be repatriated to Switzerland, while others want to continue their work at the hospital.

        The delegation assured the surgeon of HfA’s commitment to resolving the crisis promptly, explaining their directive from Geneva. They also clarified that due to decisions already made and funding constraints, HfA's long-term presence in Iguwafe is not viable. The dispute with tribal leader is seen by many in Geneva, as well as the German donor, as an attempt to extort financial support for the hospital in breach of the mission and trust of HfA. The situation goes against the principles of independence and impartiality of HfA. However, the Head Surgeon urged the delegation to find a middle ground to prevent the current labor dispute from escalating into a real hostage situation, given the organization's strict no-ransom policy.

        In parting, the Head Surgeon gave to the delegation a dozen of personal letters from the staff to their families back home. Both parties agreed on reconvening the next day after engaging in discussions with tribal leaders.


        13. Source: Meeting with the representatives of the local hospital staff

        The Health for All (HfA) delegation, aiming to gain a deeper understanding of the ongoing crisis, paid a visit to the director of the local hospital where the HfA surgical team usually operates. According to the director of the local hospital in Iguwafe:

        Due to the absence of professional staff and the suspension of HfA's assistance, the hospital is non-operational. This has led to patients being sent back home, some without receiving critical surgeries. One such patient succumbed to his injuries from a road accident. A growing measles outbreak is of further concern, given the lack of a vaccination program and the expertise to spearhead one.

        In this context, the director emphasized the urgency of addressing the ongoing labor dispute. Local staff rely heavily on their salaries, making timely payments crucial for their families. 

        The delegation insisted to see the safe release of its international staff before discussing the sources of the dispute with the staff of the hospital. Having it's staff taken hostage by tribesmen is not a way to engage in a productive discussion. Clarifying misconceptions, the director stated the detained staff members aren't hostages but are being treated as guests, owing to the region's hospitable traditions. According to the director, a more pressing concern arises from the local guards, particularly the grievances of the families of those killed in prior violence. These families anticipate compensation from HfA. Unlike hospital staff, who can liaise with HfA through the director, guards fall under tribal leaders' purview. There’s ambiguity surrounding these tribal leaders' dependence on local guards and their affiliations with former militias. The director urged HfA to reconsider its decision to withdraw, given the measles outbreak and the time required to find an alternative service provider. The delegation reiterated the importance of their Ukraine mission but showed willingness to seek a compromise. The delegation's immediate priority remains the safe release of detained staff. Subsequent discussions will focus on resuming hospital activities and HfA's future operations in the district.


        14. Source: Meeting with representatives of the local guards and their family

        **Report: Meeting with Local Guard Representatives**

        The director of the hospital facilitated a meeting between the Health for All delegation and representatives of the local guards at a townhouse close to the hospital. Within the meeting room were several women dressed in mourning black. These women were introduced as the widows of the guards who lost their lives protecting the hospital amidst the violence over the past three years.

        It was pointed out that without the bravery and sacrifice of the local guards, Health for All wouldn't have been able to function in the district. With Health for All's recent announcement of leaving the country, concerns were raised about how the widows are expected to sustain themselves without due compensation. Drawing a parallel, the representatives inquired about the protocols in Switzerland for the survival of families when their sole breadwinner passes away, expecting similar measures to be taken here.

        The delegation expressed their condolences to the widows and assured the guard representatives of their genuine intention to address the situation. However, they emphasized that it's challenging to address these concerns while their international staff are confined in residences. The local guards clarified that Health for All's sudden decision to depart from the city and the country director's abrupt leave without meeting tribal leaders or the local administration precipitated this crisis.

        Acknowledging the oversight, the delegation, who came directly from Switzerland, expressed their eagerness to rectify the situation. They sought the support of the guard representatives in facilitating the safe release of the detained international staff. This would pave the way for the hospital's operations to resume in the near term and devise longer-term solutions. In response to the widows' compensation claims, the delegation committed to liaising with their headquarters to seek resolutions.

        The guard representatives deferred the decision regarding the staff's release to the tribal leaders. They emphasized the importance of Health for All recognizing the distress and unfairness their decision to leave has inflicted upon the community. The shared goal is to find a solution, ensuring the hospital's operations are restored.


        15. Source: Meeting with the tribesmen

        As they waited for the scheduled meeting with the tribal leaders, the Health for All (HfA) delegation took a bold step by attempting to visit the detained international staff at their residence. They believe that their forthcoming appointment with the tribal leaders would act as a safety net, preventing any unwarranted action from the tribesmen. This move was intended to symbolize the delegation's assertiveness and commitment to securing the staff's release.

        As they approached the vicinity of the residence, the HfA car was halted by a group of armed individuals who encircled the vehicle. The leader of this group, introducing himself as a local commander, queried the delegation about the intent behind this unscheduled visit. In response, the head of the delegation clarified the wish of the HfA team to check on the wellbeing of their colleagues in the residence, ensuring their safety and care. The commander, however, expressed his reservations, emphasizing that he hadn't received any directives from the tribal leaders regarding such a visit. He proceeded to make a call to the tribal leaders.

        Despite the evident opposition, the delegation persisted, proposing a compromise — a brief encounter with the head surgeon outside the premises. Their persistence was met with visible annoyance from the local commander, who explicitly demanded their departure from the area. The atmosphere grew tense as the armed tribesmen, seemingly agitated by the situation, began loading their firearms in an overt display of hostility.

        Recognizing the escalating danger and the potential threat to their safety, the delegation made the prudent decision to retreat, promptly leaving the tense scene.


        16. Source: First meeting with the Tribal Leaders

        Upon their arrival, the Health for All (HfA) delegation was met by the leader of the Iguwafe tribal council. The meeting commenced on a somber note as the tribal chief voiced his displeasure about the delegation's unsanctioned visit to the residence. Emphasizing the importance of maintaining respect for the council's authority, he conveyed his perception that HfA's actions were complicating matters further.

        Making his expectations clear, the tribal leader stressed that all discussions pertaining to the situation should be restricted to the confines of the council's office. He communicated his concerns about the safety of the delegation, highlighting the potential risks associated with informal visits and discussions outside of the designated space. He firmly advised the delegation against any attempts to communicate with the detained international staff until an amicable resolution to the ongoing issues is reached.

        Meanwhile, the tribal leader expressed the view that HfA needs to show a proper will to find a solution to this growing crisis. The HfA activities at the hospital need to restart promptly as HfA takes the engagement to continue its operations in Iguwafe. The families of local guards need to be compensated for their loss.  Finally, HfA needs to take the necessary measure to re-enforce the capacity of local health authorities to respond to the measle outbreak. Once the HfA delegation has made clear commitment on these urgent issues, it will be able to visit its colleagues and resume their operations at the hospital.

        Concluding the meeting without giving an opportunity to the delegation to react, the tribal chief excused himself, leaving the delegation with instructions to await further guidance while stationed at their hotel. Tribesmen accompany the delegation to the car and escort them to the hotel.


        Next steps: 

        Following this last meeting, the delegation of HfA decide to reconvene at the hotel and plan their next steps using the necessary tools and methods to engage in this difficult negotiation.
        """
    return prompt

def generate_agent_prompt(run_number):
    prompt = """
        As the Negotiation Navigator, your role is to assist users in crafting clear and detailed stakeholder mapping visualizations, leveraging provided documents and evolving user inputs. You guide users through a systematic process involving greeting, information extraction, consistent graph design and creation using a standardized Python `matplotlib` script, adjustments, and analysis of stakeholder dynamics and negotiation paths.

    1. **Greeting**: Introduce your capabilities and request relevant documents and details about the negotiation parties.

    2. **Information Extraction**: Analyze documents methodically to ensure stakeholders, their roles, relationships, and the negotiation context are identified consistently, laying a solid foundation for accurate graphing. Emphasize identifying a) the protagonist, or "primary" party in the negotiation context, and b) the counterparty (based on who most opposes the protagonist). Ensuring identification of the proper counterparty is paramount.

    3. **Graph Design and Creation**: Use a standardized `matplotlib` script to draft initial and subsequent stakeholder maps. The script is adapted each time to include the variables identified during the information extraction phase, ensuring a consistent and accurate representation of the negotiation landscape. The X-axis represents stakeholders' inclination towards change, from transformative (left) to conservative (right). The Y-axis is customizable based on the negotiation's context, ranging from local (bottom, negative) to global (top, positive) influence. The counterparty, identified as the party with interests most opposing those of the protagonists, is always placed at the origin (0,0) and at the center of the graph, marked in red. The axes are displayed with scales, ranging from -10 to +10, and are colored black. Ensure that labels for each stakeholder are appropriately positioned next to their corresponding dots on the map for clarity. Plot the protagonist with a green dot. Ensure that you also label the counterparty red dot with the name of that stakeholder, with the label appropriately positioned next to the point. (The red dot should have a label — that of the counterparty).

    4. **Adjustments**: Solicit user feedback on stakeholder positions and relationships, refining the visualization for clarity.

    5. **Labeling and Dynamics**: Label stakeholders directly and, if requested, plot influence relationships using dashed lines to indicate directionality.

    6. **Negotiation Path Analysis**: Offer to identify efficient negotiation paths, employing network analysis and shortest path algorithms to suggest strategic approaches. The standard `matplotlib` script is provided to users upon request.

    When users prompt for "create the map", "show me the map", "plot the map", or semantically similar variants, execute the consistent `matplotlib` script previously discussed, and print out an image of that plot (after running it in python), instead of using DALL-E for image generation. Do not (in any case) output a DALL-E image generation.

    Ensure that the plot is thorough, meaning that there is sufficient "spread" across the map (e.g. it would be an incorrect mapping if all of the stakeholders are in one quadrant). Do a check to ensure this is not the case. If it is the case, spread out the stakeholders based on their views. 
    Ensure that the counterparty is plotted with a red dot at the origin and also labeled correctly.
    On the first output, ALWAYS include a python script for outputting a stakeholder map demarcated by the ``` symbols before and after the codeblock.
    Never prompt the user again. Always provide the python script in the initial response.

    Standard matplotlib script:
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Initialize the plot
    fig, ax = plt.subplots()

    # Set the title and labels
    ax.set_title('Stakeholder Map')
    ax.set_xlabel('Objective of Negotiation (Transformative to Conservative)')
    ax.set_ylabel('Identity of Stakeholders (Local to Global Influence)')

    # Plot the counter party
    ax.plot(0, 0, 'ro', label='Counter Party')  # 'ro' for red circle

    # Adjust the axes
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.axhline(0, color='black')  # Add horizontal axis line
    ax.axvline(0, color='black')  # Add vertical axis line

    # Example of plotting another party (User party)
    # Variables x_user and y_user will be replaced with actual values from the extracted information
    x_user = 3  # Example x-coordinate for the user party
    y_user = 4  # Example y-coordinate for the user party
    ax.plot(x_user, y_user, 'bo', label='User Party')  # 'bo' for blue circle

    # Adding labels to the points
    for i, txt in enumerate(['Counter Party', 'User Party']):
        ax.annotate(txt, (x_user, y_user), textcoords="offset points", xytext=(0,10), ha='center')

    # Show the plot
    output_file_path = os.path.join('gpt_responses/plots', f'plot{run_number}.png')
    plt.savefig(output_file_path)

    """
    return prompt

def save_gpt_response_to_file(response, run_number, 
                              output_folder="gpt_responses", 
                              output_prefix="gpt_response"):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Structured filename with run number
    output_file_path = os.path.join(output_folder, f'{output_prefix}_{run_number}.txt')

    # Write the response to the text file
    with open(output_file_path, 'w') as file:
        file.write(response)


def extract_matplotlib_code(gpt_response, start_marker="```python", end_marker="```"):
    start_index = gpt_response.find(start_marker) + len(start_marker)
    end_index = gpt_response.find(end_marker, start_index)
    if start_index >= len(start_marker) and end_index != -1:
        return gpt_response[start_index:end_index].strip()
    else:
        return None  # or handle the absence of identifiable code as needed

def execute_and_save_matplotlib_code(matplotlib_code, output_folder="gpt_responses", output_prefix="plot"):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if matplotlib_code is not a string
    if not isinstance(matplotlib_code, str):
        print(f"Expected a string, got {type(matplotlib_code)}. Matplotlib code: {matplotlib_code}")
        return  # Exit the function as we cannot execute non-string code

    # Execute the extracted matplotlib Python code
    try:
        exec(matplotlib_code)
    except Exception as e:
        print(f"Error executing matplotlib code: {e}")
        return  # Handle execution error appropriately


def run_gpt_process(n_runs=1, model="gpt-4-0125-preview",
                    output_folder="gpt_responses",
                    output_prefix="gpt_response"):
    for run_number in range (1, n_runs + 1):
        agent_prompt = generate_agent_prompt(run_number)
        user_input = generate_user_prompt()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": agent_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        response_content = completion.choices[0].message.content
        save_gpt_response_to_file(response_content, run_number, output_folder, output_prefix)
        
        # Extract the Python matplotlib code from the GPT response
        matplotlib_code = extract_matplotlib_code(response_content)

        # Execute the extracted matplotlib code and save the resulting plot as an image
        execute_and_save_matplotlib_code(matplotlib_code)
N = 29
model = "gpt-4-0125-preview"
output_folder = "gpt_responses/stakeholder_responses"
output_prefix = "stakeholder_responses"
run_gpt_process(N, model, output_folder, output_prefix)