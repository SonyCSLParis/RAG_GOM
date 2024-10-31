import { useState } from "react";

function App() {
  const [dataResponse, setDataResponse] = useState({});
  const [question, setQuestion] = useState("");

  async function getResponse() {
    await fetch('/get-response', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: question
      }),
    })
    .then((response) => response.json())
    .then((dataResponse) => {
      setDataResponse(dataResponse);
      console.log(dataResponse);
    });
  }

  return (
    <div>
      <div>
        <label>Votre Question</label>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button onClick={getResponse}>Submit Question</button> 
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
    </div>
  );
}

export default App;
