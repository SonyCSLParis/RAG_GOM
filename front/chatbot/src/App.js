import { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const handleSendMessage = async () => {
    if (input.trim() !== '') {
      setMessages((prevMessages) => [
        ...prevMessages,
        { type: 'text', content: input, sender: 'user' },
      ]);
      setInput('');
      try {
        const response = await fetch('/get-response', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ question: input }),
        });

        if (!response.ok) {
          throw new Error('Failed to fetch response from the server');
        }

        const data = await response.json();
        console.log(data);
        
        const normalizeString = (str) => {
          return str.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
        };
        
        const botMessages = [
          { type: 'text', content: data.answer, sender: 'bot' },
          ...data.contexte.map((context, index) => {
            let sourceImage = context.source.source
              .replace('../corpus_txt_agg', '')
              .replace('.txt', '.png');
        
            sourceImage = normalizeString(sourceImage);
        
            console.log('Chemin de l\'image normalisée:', sourceImage);

            sourceImage = `new-name/${normalizeString(sourceImage)}`;

            console.log('Chemin de l\'image avec répertoire "new-name" :', sourceImage);

        
            return {
              type: 'context',
              content: {
                sourceImage,
                pageContent: context.page_content,
              },
              sender: 'bot',
            };
          }),
        ];
        

        setMessages((prevMessages) => [...prevMessages, ...botMessages]);
      } catch (error) {
        console.error('Error fetching response:', error);
        setMessages((prevMessages) => [
          ...prevMessages,
          { type: 'text', content: 'Error fetching response.', sender: 'bot' },
        ]);
      }
    }
  };

  const renderMessage = (msg, index) => {
    switch (msg.type) {
      case 'text':
        return (
          <div
            key={index}
            className={`d-flex mb-10 ${
              msg.sender === 'user' ? 'justify-content-end' : 'justify-content-start'
            }`}
          >
            <div
              className={`p-2 rounded ${
                msg.sender === 'user'
                  ? 'bg-bgsite text-black align-self-start'
                  : 'bg-textsite text-white align-self-start'
              }`}
              style={{ maxWidth: '60%' }}
            >
              {msg.content}
            </div>
          </div>
        );
      case 'context':
        return (
          <div
            key={index}
            className="d-flex justify-content-start mb-10"
            style={{ fontStyle: 'italic', fontSize: '0.9em' }}
          >
            <div className="p-2 rounded bg-textsite text-dark" style={{ maxWidth: '60%' }}>
              <img
                src={msg.content.sourceImage}
                alt="Livre source"
                className="mx-auto mt-3"
                style={{ maxWidth: '30%' }}
              />
              {/* <p className='mt-5 text-white'>{msg.content.sourceImage}</p> */}
              <p className='mt-5 text-white'>{msg.content.pageContent}</p>
            </div>
          </div>
        );
      default:
        return null;
    }
  };
  

  return (
    <div className="h-screen w-screen d-flex align-items-end justify-content-center bg-bgsite">
      <div className="chat-container w-10/12 bg-white rounded shadow p-3 d-flex flex-column mb-4">
        <div
          className="chat-messages flex-grow-1 overflow-auto"
          style={{ maxHeight: '45rem' }}
        >
          {messages.map(renderMessage)}
        </div>
        <div className="input-group">
          <input
            type="text"
            className="form-control"
            placeholder="Posez votre question ici"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSendMessage();
            }}
          />
          <button className="btn btn-success" onClick={handleSendMessage}>
            Envoyer
          </button>
        </div>
      </div>
    </div>
  );
}
export default App;