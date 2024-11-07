import { useState } from "react";
import ChatBot from "react-chatbotify";

function App() {
  const [dataResponse, setDataResponse] = useState({});
  const [question, setQuestion] = useState("");

  async function getResponse(question) {
    const response = await fetch('/get-response', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: question
      }),
    });
    
    const dataResponse = await response.json();
    setDataResponse(dataResponse);
    console.log(dataResponse);

    return dataResponse.answer;
  }

  const flow = {
    start: {
      message: "Votre question",
      path: "model_loop",
    },
    model_loop: {
      message: async (params) => {
        return await getResponse(params.userInput);
      },
      path: "model_loop"
    },
  };

  return (
    <div>
      <div>
        <label>Votre Question</label>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button onClick={() => getResponse(question)}>Submit Question</button> 
      </div>
      <div>
        {dataResponse.contexte && (
          <div>
            <h3>Contexte :</h3>
            <p>{dataResponse.contexte}</p>
          </div>
        )}
        {dataResponse.input && (
          <div>
            <h3>Question :</h3>
            <p>{dataResponse.input}</p>
          </div>
        )}
        {dataResponse.answer && (
          <div>
            <h3>Réponse :</h3>
            <p>{dataResponse.answer}</p>
          </div>
        )}
      </div>
      <ChatBot flow={flow}/>
    </div>
  );
}

export default App;
