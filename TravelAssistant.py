from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START

import os
import requests
from typing import TypedDict, List
from dotenv import load_dotenv
from datetime import datetime, timedelta
from serpapi import GoogleSearch

from RAG import RAG

class TravelAssistant:
    def __init__(self):
        load_dotenv()

        self.rag = RAG()
        self.rag.filter_flag = False
        self.rag.set_system_prompt("You are a helpful travel agent acting as an assistant for question-answering tasks. "
                                    "Use the following pieces of retrieved context to answer the question. "
                                    "If you don't know the answer, just say that you don't know."
                                    "\n\n{context}")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        self.vector_store_topics = []

        self.initial_message = "Hello there! I'm here to help you with all of your travel needs. Ask me to find you flights, recommend things to do on your vacation, or whatever you need assistance with!"
        self.rag.store[self.rag.directory.name] = ChatMessageHistory()
        self.rag.store[self.rag.directory.name].add_ai_message(AIMessage(self.initial_message))

        self.conversation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful travel agent providing travel advice and making friendly conversation with "
                 "the user. You have the capability to find flights, hotels, restaurants, and events. You are also capable of "
                 "performing web search to find relevant travel information. Refuse to discuss outside topics unrelated to travel "
                 "or conversation."),
                 MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        self.conversation_chain = RunnableWithMessageHistory(
            self.conversation_prompt | self.llm,
            self.rag.get_session_history, # RAG and base LLM share conversation history
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="output",
        ).with_config(config = {
                "configurable": {"session_id": self.rag.directory.name}
        })

        self.google_query_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant. You will be given a user prompt, and you will "
                 "create a Google search query to retrieve outside knowledge helpful in answering the prompt."),
                ("human", "User prompt: {input}\nGoogle search query: ")
            ]
        )
        self.google_query_chain = self.google_query_prompt | self.llm

        self.rag_term_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant who receives a Google search query and outputs the topic which the query was about. "),
                ("human", "Google search query: {input}\Topic: ")
            ]
        )
        self.rag_term_chain = self.rag_term_prompt | self.llm

        self.find_location_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant. You will be given a user prompt and you will extract the given location in regards "
                 "to the {item} being discussed. Output just the location, nothing else."),
                ("human", "User prompt: {input}\Location: ")
            ]
        )
        self.find_location_chain = self.find_location_prompt | self.llm

        self.event_period_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant. You will be given a user prompt containing information about dates regarding potential events. Classify "
                 "the referenced dates as either 'month' meaning this month given the current date: {date}, or 'next_month'. Output only this classification."),
                 ("human", "User prompt: {input}\n\nDate classificaton: ")
            ]
        )
        self.event_period_chain  = self.event_period_prompt | self.llm

        self.flight_info_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant. You will be given a user prompt containing information about desired flights, "
                 "and you will extract the start location, the destination, the departure date, and the return date. You will output each "
                 "of those on seperate lines with no extra formatting, in the order given. Output the dates in YYYY-MM-DD format."),
                ("human", "User prompt: {input}\n)Output: ")
            ]
        )
        self.flight_info_chain = self.flight_info_prompt | self.llm

        self.present_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant. You will be given some data about either flights, hotels, restaurants, or events. Given the data, "
                 "you will output a final response incorporating all of the data in a seamless and natural way to show our findings to the user."),
                ("human", "Data:\n{input}\n\n\nOutput:\n")
            ]
        )
        self.present_chain = self.present_prompt | self.llm
        
        self.fully_answered_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant. You will be given a question and a response. If the response fully answers the question, output 'yes', otherwise, output the missing information."
                 "The answer does not need to be complicated or verbose. If it answers the question in its most basic form, consider it answered."
                 "Example: Question: When will you be departing and returning, and where are you flying from?\nResponse: I will be leaving 7/13/2024.\nOutput:\ndeparture date and start location"),
                ("human", "Question: {q}\n Response: {r}\nOutput:\n")
            ]
        )
        self.fully_answered_chain = self.fully_answered_prompt | self.llm

        self.split_prompt = ChatPromptTemplate.from_messages( # splits prompts into individual tasks which agents can perform; few-shot prompting to improve accuracy
            [
                ("system", '''You are an assistant to a travel agent.
You will be given a user prompt and will need to identify the individual tasks needing to be handled by the travel agent.
A task consists of one of the following:
Having conversation through an LLM
Accessing a vector store (which {veccontext}) to assist answering user questions
Utilizing a web search to access outside information not in the vector store to help the user
Accessing a hotel API to provide hotel information: requires location
Acessing a flights API to provide flight information: requires start location, destination, departure date, and return date
Acessing a restaurant API to provide restaurant information: requires location
Accessing an events API to provide information about various events: requires location and date\n\n
You should send any topics unrelated to travel or polite conversation to "llm".
The web search should be used absolutely whenever outside knowledge could be useful in asnwering the user. However, if the required knowledge is in the vector store, the vector store should be used instead.
You will be given a user prompt.
For each individual task write a prompt to the agent, containing the information and prompting needed for that task and that task only.
The output should be the new prompts preceded by their type of task ("hotels", "flights", "restaurants", "events", "vector_store", "web_search", or "llm"), and printed on seperate lines.
These prompts should contain reformulations of what the user said, as opposed to answers, so that the agent can respond (eg. user prompt: "Hi, I'm Daniel", output: "llm: Hi, I'm Daniel").
You may also be given a chat history to provide context for the user prompt. It's critical that you don't create tasks out of the history. It's only for context.
If all the required information for an API task is in the provided prompt or context, then the prompt for that task should be only the required information as shown in the examples which will follow.
Otherwise, API tasks for which not all of the required information is present must be preceded by "moreinfo [tasktype]", instead of just their type of task, followed by a prompt asking the user to provide the missing info, as shown in the examples.
Put all "moreinfo"s after the rest of the prompts.
To emphasize, API tasks which are missing required info must be preceded by "moreinfo [tasktype]", followed by a prompt for the user to provide the missing info.
To repeat, do not respond to the user in your output. Your job is to synthesize what the user said so that the agent can respond.
Make sure that all tasks which require additional info are labeled "moreinfo". \n
Example 1:
Context:
User: Hi, I'm Daniel. I'm visiting Vienna soon. Can you tell me some good restaurants there?
AI: Certainly! Steirereck and Lugeck are both famous restaurants known for their world-class Austrian food.\n
User prompt: Can you get me tickets to fly there and also find out if anything interesting will be going on on 7/20?\n
Output:
moreinfo flights: Where will you be flying from, and when were you thinking of leaving and returning?
events: Vienna, 07/20/2024\n
Example 2:
User prompt: Hi, I'm Daniel. Can you find me hotels and nice restaurants?\n
Output:
llm: Hi, I'm Daniel. Can you find me hotels and restaurants.
moreinfo hotels: Where were you thinking of staying?
moreinfo restaurants: Where were you thinking about looking for restaurants?\n
Example 3:
Context:
User: Hi, I'm Daniel. I'm visiting Vienna soon. Can you tell me some good restaurants there?
AI: Certainly! Steirereck and Lugeck are both famous restaurants known for their world-class Austrian food.\n
User prompt: Can you also find me hotels and tell me if I need a visa to go to there from the US?\n
Output:
web_search: Do you need a visa to go to Austria from US.
hotels: Vienna'''),
                ("human", "{input}\nOutput:")
            ]
        )
        self.split_chain = self.split_prompt | self.llm

        class GraphState(TypedDict):
                prompts: List[str]
                user_prompt: str
                index: int
                end: bool
                new_hist: List[BaseMessage]
        self.workflow = StateGraph(GraphState)

        self.workflow.add_node("hotels", self.hotels)
        self.workflow.add_node("flights", self.flights)
        self.workflow.add_node("restaurants", self.restaurants)
        self.workflow.add_node("events", self.events)
        self.workflow.add_node("rag", self.call_rag)
        self.workflow.add_node("web_search", self.web_search)
        self.workflow.add_node("llm", self.call_llm)
        self.workflow.add_node("splitter", self.split)
        self.workflow.add_node("route", self.route)
        self.workflow.add_node("input", self.get_input)
        self.workflow.add_node("cleanup", self.cleanup)

        self.workflow.add_edge(START, "input")
        self.workflow.add_conditional_edges(
            "input",
            self.should_continue, 
            {
                "yes": "splitter",
                "no": "cleanup"
            }
        )
        self.workflow.add_edge("splitter", "route")

        self.workflow.add_conditional_edges(
            "route",
            self.route_question,
            {
                "hotels": "hotels",
                "flights": "flights",
                "restaurants": "restaurants",
                "events": "events",
                "vector_store": "rag",
                "web_search": "web_search",
                "llm": "llm",
                "next": "input",
            }
        )

        self.extract_prompt = lambda x: x[(x.index(":") + 1):].strip()

        self.workflow.add_edge("web_search", "rag")
        self.workflow.add_edge("rag", "route")
        self.workflow.add_edge("hotels", "route")
        self.workflow.add_edge("flights", "route")
        self.workflow.add_edge("restaurants", "route")
        self.workflow.add_edge("events", "route")
        self.workflow.add_edge("llm", "route")

        self.workflow.add_edge("cleanup", END)

        self.app = self.workflow.compile()

    def veccontext(self): # give context about the current information in the vector store
        if self.vector_store_topics == []:
            return "is currently empty and should be ignored"
        return "should be used regarding following topics: " + ", ".join(self.vector_store_topics)
    
    def route(self, state):
        return state
    
    def get_input(self, state): # wait for user input
        prompt = input("Prompt: ").strip()
        
        return {"user_prompt":  prompt, "end": prompt == "stop", "new_hist": [HumanMessage(prompt)], "prompts": [], "index": 0} # reset state

    def should_continue(self, state):
        if state["end"]:
            return "no"
        return "yes"
    
    def format_chat_hist(self, new_hist):
        chat_history = ""
        print(new_hist)
        for message in self.rag.store[self.rag.directory.name].messages + new_hist:
            if isinstance(message, AIMessage):
                chat_history += "AI: "
            else:
                chat_history += "User: "
            chat_history += message.content.strip() + "\n"
        return chat_history.strip()

    def split(self, state): # split current tasks into individual tasks
        chat_history = "Context (Do not make tasks out of this, it is only for context):\n" + self.format_chat_hist([])
        chat_history += "\nUser prompt (Make tasks out of this): " + state["user_prompt"]

        split_output = self.split_chain.invoke({"veccontext": self.veccontext(), "input": chat_history}).content.strip()
        print(split_output, end="\n\n")

        lines = split_output.splitlines()
        if lines[0].lower().strip() == "output:":
            del lines[0]

        return {"prompts": lines, "index": 0}
    
    def route_question(self, state): # route each prompt to the proper agent based on its label or go to next user input if done
        if state["index"] >= len(state["prompts"]):
            self.rag.store[self.rag.directory.name].messages += state["new_hist"]
            return "next"
        t_type = state["prompts"][state["index"]].split(':')[0]
        if ' ' in t_type:
            return t_type.split(' ')[1]
        return t_type

    def query_info(self, query): # ask for more info from the user
        print(query)
        response = input("Prompt: ").strip()
        return response

    def hotels(self, state):
        prompt = self.extract_prompt(state["prompts"][state["index"]])
        response = self.rag.rephraser.invoke({"chat_history": self.rag.store[self.rag.directory.name].messages + state["new_hist"], "input": state["user_prompt"]}).strip()
        if state["prompts"][state["index"]].split(' ')[0] == "moreinfo":
            response = self.query_info(prompt) 
            state["new_hist"] += [AIMessage(prompt), HumanMessage(response)]
            fully_answered = self.fully_answered_chain.invoke({"q": prompt, "r": response}).content.strip().lower()
            if fully_answered != "yes":
                print("I can not find hotels without knowing " + fully_answered)
                return {"index": state["index"] + 1, "new_hist": state["new_hist"]}
        location = self.find_location_chain.invoke({"item": "hotels", "input": self.format_chat_hist(state["new_hist"]) + response}).content.strip()
        
        filler_checkin = (datetime.today() + timedelta(days=3)).strftime("%Y-%m-%d")
        filler_checkout = (datetime.today() + timedelta(days=4)).strftime("%Y-%m-%d")

        url = "https://booking-com.p.rapidapi.com/v1/hotels/locations"
        querystring = {"name": location, "locale":"en-us"}
        headers = {
            "x-rapidapi-key": os.getenv("X_RAPIDAPI_KEY"),
            "x-rapidapi-host": "booking-com.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers, params=querystring).json()

        url = "https://booking-com.p.rapidapi.com/v1/hotels/search"
        querystring = {
            "checkout_date": filler_checkout,
            "order_by": "popularity",
            "filter_by_currency": "USD",
            "room_number": "1",
            "dest_id": response[0]["dest_id"],
            "dest_type": response[0]["dest_type"],
            "adults_number": "1",
            "checkin_date": filler_checkin,
            "locale": "en-us",
            "units": "metric"}
        headers = {
            "x-rapidapi-key": os.getenv("X_RAPIDAPI_KEY"),
            "x-rapidapi-host": "booking-com.p.rapidapi.com"
        }
        results = requests.get(url, headers=headers, params=querystring).json()["result"]
        data = "Hotels:\n"
        for x in range(min(len(results), 4)):
            result = results[x]
            hotel_name = result["hotel_name_trans"]
            data += hotel_name + " -\n"
            review_score = result["review_score"]
            data += "Review score: " + str(review_score) + "\n"
            photo_url = result["max_photo_url"]
            booking_url = result["url"]
            min_price_per_night = result["min_total_price"]
            data += "Cost: from $" + str(min_price_per_night) + " per night\n\n"
            print("HOTEL", x, hotel_name, review_score, booking_url, min_price_per_night)

        return {"index": state["index"] + 1, "new_hist": state["new_hist"]}

    def flights(self, state):
        prompt = self.extract_prompt(state["prompts"][state["index"]])
        response = self.rag.rephraser.invoke({"chat_history": self.rag.store[self.rag.directory.name].messages + state["new_hist"], "input": state["user_prompt"]}).strip()
        if state["prompts"][state["index"]].split(' ')[0] == "moreinfo":
            response = self.query_info(prompt)
            state["new_hist"] += [AIMessage(prompt), HumanMessage(response)]
            print(prompt, response)
            fully_answered = self.fully_answered_chain.invoke({"q": prompt, "r": response}).content.strip().lower()
            if fully_answered != "yes":
                print("I can not find flights without knowing " + fully_answered)
                return {"index": state["index"] + 1, "new_hist": state["new_hist"]}
        data = self.flight_info_chain.invoke({"input": self.format_chat_hist(state["new_hist"]) + response}).content.strip()
        start = data.split("\n")[0]
        destination = data.split("\n")[1]
        depart_date = data.split("\n")[2]
        return_date = data.split("\n")[3]

        url = "https://booking-com15.p.rapidapi.com/api/v1/flights/searchDestination"
        querystring = {"query": start}
        headers = {
            "x-rapidapi-key": os.getenv("X_RAPIDAPI_KEY"),
            "x-rapidapi-host": "booking-com15.p.rapidapi.com"
        }
        start_code = requests.get(url, headers=headers, params=querystring).json()["data"][0]["code"]
        querystring = {"query": destination}
        dest_code = requests.get(url, headers=headers, params=querystring).json()["data"][0]["code"]

        params = {
            "engine": "google_flights",
            "departure_id": start_code,
            "arrival_id": dest_code,
            "outbound_date": depart_date,
            "return_date": return_date,
            "currency": "USD",
            "hl": "en",
            "api_key": os.getenv("SERP_API_KEY")
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        print(results)
        url = results["search_metadata"]["google_flights_url"]
        flights = results["best_flights"][0]
        print(flights)

        return {"index": state["index"] + 1, "new_hist": state["new_hist"]}

    def restaurants(self, state):
        prompt = self.extract_prompt(state["prompts"][state["index"]])
        response = self.rag.rephraser.invoke({"chat_history": self.rag.store[self.rag.directory.name].messages + state["new_hist"], "input": state["user_prompt"]}).strip()
        if state["prompts"][state["index"]].split(' ')[0] == "moreinfo":
            response = self.query_info(prompt)
            state["new_hist"] += [AIMessage(prompt), HumanMessage(response)]
            fully_answered = self.fully_answered_chain.invoke({"q": prompt, "r": response}).content.strip().lower()
            if fully_answered != "yes":
                print("I can not find restaurants without knowing " + fully_answered)
                return {"index": state["index"] + 1, "new_hist": state["new_hist"]}
        location = self.find_location_chain.invoke({"item": "restaurants", "input": self.format_chat_hist(state["new_hist"]) + response}).content.strip()
        print(location)
        url = "https://tripadvisor16.p.rapidapi.com/api/v1/restaurant/searchLocation"
        querystring = {"query":location}
        headers = {
            "x-rapidapi-key": os.getenv("X_RAPIDAPI_KEY"),
            "x-rapidapi-host": "tripadvisor16.p.rapidapi.com"
        }
        loc_id = requests.get(url, headers=headers, params=querystring).json()["data"][0]["locationId"]
        url = "https://tripadvisor16.p.rapidapi.com/api/v1/restaurant/searchRestaurants"
        querystring = {"locationId": loc_id}
        headers = {
            "x-rapidapi-key": os.getenv("X_RAPIDAPI_KEY"),
            "x-rapidapi-host": "tripadvisor16.p.rapidapi.com"
        }
        responses = requests.get(url, headers=headers, params=querystring).json()["data"]["data"]
        print(responses)
        with_img = 0
        for result in responses:
            if "heroImgUrl" in result:
                with_img += 1
                photo_url = result["heroImgUrl"]
                name = result["name"]
                tags = ", ".join(result["establishmentTypeAndCuisineTags"])
                print(name, tags, photo_url)
            else:
                continue
            if with_img >= 3:
                break
        
        return {"index": state["index"] + 1, "new_hist": state["new_hist"]}

    def events(self, state):
        prompt = self.extract_prompt(state["prompts"][state["index"]])
        response = self.rag.rephraser.invoke({"chat_history": self.rag.store[self.rag.directory.name].messages + state["new_hist"], "input": state["user_prompt"]}).strip()
        if state["prompts"][state["index"]].split(' ')[0] == "moreinfo":
            response = self.query_info(prompt)
            state["new_hist"] += [AIMessage(prompt), HumanMessage(response)]
            fully_answered = self.fully_answered_chain.invoke({"q": prompt, "r": response}).content.strip().lower()
            if fully_answered != "yes":
                print("I can not find events without knowing " + fully_answered)
                return {"index": state["index"] + 1, "new_hist": state["new_hist"]}
        location = self.find_location_chain.invoke({"item": "event(s)", "input": self.format_chat_hist(state["new_hist"]) + response}).content.strip()
        period = self.event_period_chain.invoke({"date": datetime.today().strftime("%m/%d/%y") + " (mm/dd/yyyy)", "input": self.format_chat_hist(state["new_hist"]) + response}).content.strip()
        print(location, period)
        query = self.google_query_chain.invoke({"input": response + " . " + location}).content.strip()
        print(query)
        url = "https://real-time-events-search.p.rapidapi.com/search-events"
        querystring = {"query": query, "is_virtual":"false"}
        headers = {
            "x-rapidapi-key": os.getenv("X_RAPIDAPI_KEY"),
            "x-rapidapi-host": "real-time-events-search.p.rapidapi.com"
        }
        summary = ""
        results = requests.get(url, headers=headers, params=querystring).json()["data"]
        
        for x in range(min(len(results), 3)):
            result = results[x]
            date = result["start_time"][:10]
            summary += str(x + 1) + ": " + result["name"] + ". " + result["link"] + " " + date + "\n"
        print(summary) 
        
        return {"index": state["index"] + 1, "new_hist": state["new_hist"]}

    def web_search(self, state): # call rag web_search and then continue to rag node
        prompt = self.extract_prompt(state["prompts"][state["index"]])
        search_query = self.google_query_chain.invoke({"input": prompt}).content.strip()
        self.rag.web_search(search_query, 5)
        self.vector_store_topics.append(self.rag_term_chain.invoke({"input": search_query}).content.strip())
        return state

    def call_rag(self, state):
        prompt = self.extract_prompt(state["prompts"][state["index"]])
        output = self.rag.query(prompt)
        del self.rag.store[self.rag.directory.name].messages[-2:] # get rid of intermediary prompts
        return {"index": state["index"] + 1, "new_hist": state["new_hist"] + [AIMessage(output)]}

    def call_llm(self, state):
        prompt = self.extract_prompt(state["prompts"][state["index"]])
        output = self.conversation_chain.invoke({"input": prompt})
        print(output)
        del self.rag.store[self.rag.directory.name].messages[-2:] # get rid of intermediary prompts
        return {"index": state["index"] + 1, "new_hist": state["new_hist"] + [AIMessage(output)]}

    def cleanup(self, state):
        self.rag.cleanup()
        return state
    
    def cmd_run(self):
        print(self.initial_message)
        self.app.invoke({"end": False, "new_hist": []}, config={"recursion_limit": 5000})
