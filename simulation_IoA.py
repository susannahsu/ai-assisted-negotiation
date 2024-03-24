import os
import json
from openai import OpenAI
from dotenv import load_dotenv

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
    Here's the background and other information of the case.

    counter party: camp authorities
    user party: FWB

    Background:
    Food Without Borders Assisting Refugees in Alpha

    FWB is an international humanitarian organization dedicated to providing food assistance to populations affected by crises worldwide. Operating in over 70 countries, FWB's overarching mission is to save lives and restore human dignity through the provision of food aid. FWB is planning the distribution of urgently needed food aid within a refugee camp situated in Country Alpha. 

    The UN Humanitarian Coordinator in Alpha has issued a press release expressing profound concern for the plight of over 160,000 forcibly displaced refugees, a consequence of acute violence in Omega, a neighboring country, over the past few months. According to the statement, these refugees have settled in multiple camps in proximity to the capital. The camps are receiving a continuous influx of new arrivals, with reports from local church activists highlighting a majority of newcomers as children and women facing severe nutritional deficiencies, along with inadequate shelter and access to clean water. The overall sanitary and hygiene conditions within the camps are dire, with healthcare services being limited to mobile clinics operated by an NGO called Health for All. Insights from contacts within the refugee population indicate that the displaced individuals are confined to the camps under the watch of local guards and require official permission to leave.

    The primary camp, housing over 160,000 displaced individuals, is located near the National Hero Roundabout, approximately 35 km north of the capital. This camp is overseen by local authorities with established links to armed militias actively participating in hostilities against Omega's government. These militias have been implicated in targeting civilian populations in Omega and attacking refugees originating from Omega. MASHA, a local NGO, has reported that a significant portion of the primary camp's security guards are members of the armed militia, and they have been observed imposing stringent controls over the camp's population, occasionally engaging in extortion and violence against the most vulnerable refugees.

    FWB has offered to provide emergency food rations to the recently arrived refugee families, and the camp authorities have agreed to accept FWB's food aid. However, the camp authorities are requiring that FWB hires local guards to assist in the food ration distribution process. The camp authorities argue that the guards' duties during food distribution extend beyond their security roles and should be compensated similarly to other day laborers. Importantly, the camp authorities will not permit anyone else to work on behalf of FWB within the camp.

    Additionally, the camp authorities anticipate that the local guards should receive compensation in the form of food rations for their services. They argue that the families of these guards are also grappling with food insecurity, and cash payments hold limited utility in the region due to exorbitant local market food prices. Food rations are increasingly becoming the sole acceptable means of compensation. However, this request contradicts FWB's policies about in-kind food payments due to concerns about the risk of the food aid being diverted from its intended beneficiaries to be resold at market price. In response, the Camp Commander has noted that many refugees are already trading their rations on the camp's market in exchange for phone SIM cards.

    As a humanitarian organization, FWB is steadfast in its commitment to providing emergency food aid to the refugee population as swiftly and impartially as possible. FWB representatives are deeply apprehensive that entrusting local guards with the food distribution could potentially lead to further exploitation of the refugees. The demands set forth by the camp authorities appear to be at odds with FWB's core principles of neutrality, impartiality, and independence.

    The local team of FWB is now tasked with devising a negotiation strategy to engage with the camp authorities. The objective of this negotiation with the Camp Commander is to secure its presence and establish the terms of operation within the refugee camp, aiming for optimal food distribution results while minimizing any adverse impact on FWB as an organization and the refugee population as beneficiaries. In the absence of a prompt agreement, the lives of thousands of refugees will be at risk.


    Additional Information and Notes About the Situation

    Source: UN report: 

    According to the UN Humanitarian Coordinator, large numbers of newly arrived refugees are facing food shortages. The number of refugees is increasing day by day. There are now over 160,000 people in the camp near the National Hero Roundabout.

    Source: Minutes of Meeting (MoM) with the Camp Commander: 

    FWB seeks to identify refugees experiencing food insecurity; however, the camp figures are outdated. FWB intends to conduct a nutritional assessment, which may take approximately one week due to the population’s size. This assessment is crucial before initiating food distribution, as it aligns with the German government's stipulation. The German government, a significant donor to FWB, insists on a comprehensive nutritional assessment to ensure food aid goes to those truly in need, rather than being diverted by local armed militias.

    In the meeting, the Camp Commander asserted that existing data collected by the Camp Administration regarding the population's nutritional status is sufficient. He deems a new technical assessment unnecessary and requires FWB proceed swiftly with a three-month food ration distribution to alleviate pressure on refugee families.

    Source: MoM with MASHA, a local charitable organization: 

    According to early findings gathered by MASHA within the camp, a majority of refugees are in dire need of food assistance. There are concerns of aid diversion due to the camp guards' affiliation with the local militia, known to exploit camp residents. Instances of violence and threats by guards against refugees who refuse to provide money or food rations have been documented. The guards exercise significant control over the camp and its population.	

    MASHA reports that refugees are essentially confined to the camp, requiring authorization from camp authorities to leave. They are unable to seek employment outside the camp and depend entirely on humanitarian food assistance. Some refugees resort to selling their food rations to obtain phone SIM cards and other necessities from local merchants, who, in turn, sell the rations in the villages and nearby communities surrounding the camp.

    MASHA informs FWB that there is growing pressure on the Camp Commander to secure renewed food assistance, as camp stores are nearly depleted. Without an immediate influx of food rations from humanitarian organizations, the Camp Commander lacks the resources to pay the guards. Moreover, without food supplies, the refugee population may attempt to leave the camp at the National Hero Roundabout, heading closer to the capital. This would be disastrous for the Camp Commander, the local militia, and the village that relies on camp-related activities. The camp was strategically established to prevent refugees from settling in the vicinity of the capital.

    Source: MoM with Camp Commander: 

    The Camp Commander acknowledges the need for relief assistance to reach refugees but emphasizes the importance of a clear distribution plan. He insists that guards must always accompany FWB representatives within the camp due to security concerns. However, FWB, based on its principles of neutrality, impartiality, and independence, requests the freedom for its staff to move within the camp and engage with refugees without guard presence. This request does not sit well with the Camp Commander.

    Source: MoM with the Camp Administrator: 

    The Camp Administrator estimates that FWB will require 200 daily laborers for food ration distribution within the camp. A list of potential hires, predominantly consisting of guards and their family members from the local community, has been provided. FWB accepts the list but asserts that all names must undergo vetting by FWB administration. The organization does not intend to hire guards or their relatives, citing concerns about aid diversion and maintaining neutrality. Ideally, FWB wishes to employ individuals from the local community surrounding the camp. FWB commits to contacting the local mayor to explore recruitment options for 200 daily laborers. Although acknowledging the importance of involving the local community, the Camp Administrator appears displeased with this plan.

    Source: MoM with the Camp Commander: 

    Camp Authorities has informed FWB that it does not allow FWB to hire anybody else but guards. FWB informs the Camp authorities that it cannot hire guards because they are members of the local militia – that is party to the conflict. To hire the guards would be a violation to the rules of neutrality and independence of FWB.

    Source: MoM with the Mayor of the local village: 

    The Mayor informs FWB that while food insecurity prevails in the region, there is no severe hunger. The local economy heavily relies on job opportunities within the camp. Many young men are part of the local militia, and their families benefit when their sons serve in this capacity. Upon their return from fighting on the frontlines, their benefits may include employment as guards at the camp, with compensation in the form of food rations.

    The Mayor confirms that there are very few consumer goods available in the market, so most payment to the staff of the camp take place in the form of food rations that are then sold or exchanged for other necessities at the market of the village. One can see the logos of humanitarian organizations in the packaging of the goods sold at the market. The Mayor believes that FWB will probably need to pay its daily laborer in food rations as there in not much to buy with money at the market. FWB explains to the Mayor that it prefers to pay its employees and daily laborers in cash to avoid encouraging the diversion of food assistance from the refugees. According to the principles of FWB and agreements with the donors, all the FWB food should go to the refugees. The Mayor recommends to FWB to find a practical solution to the issue of hiring and compensating local guards. It is most unlikely that the Camp Commander would allow other people to work in the camp as the guards depend on the food provided by international NGOs to feed their families.

    Source: MoM with the Minister of Defense of Alpha

    In a tense meeting between FWB representatives and the Minister of Defense of Alpha, it became apparent that the Alpha government views all refugees from Omega as potential terrorists infiltrating its territory. Alpha mandates strict regulation of these individuals within camps. Alpha perceives FWB's role as supporting the government of Alpha and the Camp Administration. Furthermore, Alpha does not consider them as “refugees” in a legal sense and is ready to deport anyone who does not comply with the laws of Alpha.

    Source: Press Statement of the Council of Omega Abroad

    The Council expresses concern over the influx of refugees from Omega into Alpha. It condemns the violence against Omega's refugees, perpetrated by the government and its affiliated "terror brigades." The Council calls on the international community to provide urgent assistance and protection to these refugees in Alpha. The Council encourages refugees to leave Alpha, which it does not consider a safe haven for Omega's people.

    Source: MoM with Health for All

    During a meeting between FWB representatives and Health for All, an NGO operating mobile clinics in the camp, Health for All expresses deep concern for the well-being and dignity of the camp's population. Numerous threats, including disease, malnutrition, and violence, endanger refugees. Health for All urges FWB to adopt a pragmatic approach to operate effectively within the camp. Health for All discloses that it has been working with and compensating guards around mobile clinics as a strategy to manage the population accessing healthcare. Furthermore, it offers free medical services to guards since many of the them have suffered injuries on the frontlines.

    Source: Local press

    Local press reports have taken a critical stance toward refugees, emphasizing the continuous influx of refugees and a rise in crimes around the camps. The press supports the government's efforts to confine refugees within camps where they can receive assistance. It calls on the international community to provide equitable aid to the local communities surrounding the camps, which have also suffered due to the conflict and regional tensions.

    Source: MoM with local church activists

    Local church activists express growing concerns about the increasing influence of former militia members and their families in the community. They perceive these ex-fighters from the frontlines as introducing negative behaviors and endorsing abusive practices. They highlight refugees as the primary victims of these abuses within the camps. Local community members have also fallen victim to abuses by former militia members. Local church activists support FWB's decision to refrain from further empowering militia members and local guards by hiring them and compensating them with food rations.

    Source: MoM with the leaders of the Militia

    In an unexpected meeting between FWB representatives and militia leaders, the leaders express their frustration with what they perceive as FWB's discrimination against local guards, who they believe deserve food assistance just as much as anyone else. These guards have served their country and are deemed more deserving of aid than refugees who have merely fled conflict. The militia leaders indicate their intent to closely monitor FWB, considering it a source of support for numerous spies and infiltrators within the camp. These individuals are not refugees but rather viewed as terrorists.

    Source: MoM with the leaders of the local guards

    Leaders of the local guards visit FWB's camp office and present a list of urgently needed food items for themselves and their families. They emphasize that they will not authorize their members to work in the camp unless their requested assistance is provided.

    Source: MoM with refugee leaders

    Refugee leaders within the camp meet with FWB's camp office head. They express frustration with the international community's perceived inaction regarding their situation. They feel trapped by the local guards and militia and call upon the world to intervene and free them from ongoing suffering, particularly women and children. They strongly oppose the idea of local guards overseeing humanitarian food distribution, given the abuses they've endured at the hands of the militia. Refugee leaders are critical of FWB and other charitable organizations for their perceived lack of accountability toward camp beneficiaries.
    """
    return prompt

def generate_agent_prompt():
    prompt = """
    You are a highly structured and detail-oriented assistant specializing in negotiation analysis. Based on the provided information about a high-stakes situation, your task is to analyze and summarize the critical information to fill out an Island of Agreement (IoA) table and recommend negotiation strategies. The IoA table consists of four categories: contested facts (facts to the clarified with factual evidence), agreed facts (points of agreement to start the dialogue), convergent norms (points to be underlined as convergent values), and divergent norms (points of divergence on norms to be negotiated). 
    
    Here is how you can populate the Island of Agreement table:
    Step 1 - Sorting and qualifying elements arising in a negotiation environment. Be sure to differentiate between negotiation facts and norms. Facts include number and features of the beneficiary population, location of this population, technical terms of the assistance programs (time, date, mode of operation), nutritional and health status of the population, etc. Norms include right of access to the beneficiary population, obligations of the parties, legal status of the population, and priority of the operation, etc.
    Step 2 - Recognizing which areas of the conversation are most/least promising in the establishment of a relationship and which concrete issues will need to be negotiated with the counterparts.
    Step 3 - Elaborating a common understanding with the counterpart on the point of departure of the discussion while underlining the specific objectives of the negotiation process.

    For each category, please provide a structured list of items. Follow this structured response format:
    - Contested Facts: Fact1, Fact2, ...
    - Agreed Facts: Fact1, Fact2, ...
    - Convergent Norms: Norm1, Norm2, ...
    - Divergent Norms: Norm1, Norm2, ...

    Based on the analysis, please also list recommendations on what to prioritize and what to avoid in the negotiation, formatted as follows:
    - Prioritize: Item1, Item2, ...
    - Avoid: Item1, Item2, ...

    Please provide the response in a concise, structured format that can be easily parsed. User will provide you with background and details of the case.
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

def run_gpt_process(n_runs=1, model="gpt-4-0125-preview",
                    output_folder="gpt_responses",
                    output_prefix="gpt_response"):
    for run_number in range (1, n_runs + 1):
        agent_prompt = generate_agent_prompt()
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

N = 2
model = "gpt-4-0125-preview"
output_folder = "gpt_responses/IoA_response"
output_prefix = "IoA_response"
run_gpt_process(N, model, output_folder, output_prefix)