import { useState, useEffect, useRef } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (input.trim() !== '') {
      const userMessage = { type: 'text', content: input, sender: 'user' };
      setMessages((prevMessages) => [...prevMessages, { group: [userMessage] }]);
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

        const botResponse = {
          type: 'text',
          content: data.answer,
          sender: 'bot',
        };

        const contextMessages = data.contexte.map((context) => {
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
        });

        setMessages((prevMessages) => [
          ...prevMessages,
          { group: [botResponse, ...contextMessages] },
        ]);
      } catch (error) {
        console.error('Error fetching response:', error);
        setMessages((prevMessages) => [
          ...prevMessages,
          { group: [{ type: 'text', content: 'Error fetching response.', sender: 'bot' }] },
        ]);
      }
    }
  };

  return (
    <div className="h-screen w-screen d-flex align-items-end justify-content-center bg-bgsite">
      <div className="chat-container w-10/12 bg-white rounded shadow p-3 d-flex flex-column mb-4">
        <div
          className="chat-messages flex-grow-1 overflow-auto h-full"
          style={{ maxHeight: '45rem' }}
        >
          {messages.map((messageGroup, groupIndex) => (
            <div key={groupIndex} className="message-group mb-4 d-flex align-items-start">
              <div className="d-flex flex-column max-w-lg">
                {messageGroup.group
                  .filter((msg) => msg.type === 'text')
                  .map((msg, index) => (
                    <div
                      key={index}
                      className={`d-flex ${
                        msg.sender === 'user' ? 'mt-3' : 'mb-3'
                      }`}
                    >
                      <div
                        className={`p-3 rounded ${
                          msg.sender === 'user'
                            ? 'bg-bgsite text-black align-self-start'
                            : 'bg-textsite text-white align-self-start'
                        }`}
                      >
                        {msg.content}
                      </div>
                    </div>
                  ))}
              </div>
              <div
                className="d-flex flex-column max-w-xl ml-10 overflow-auto"
                style={{ maxHeight: '20rem' }}
              >
                {messageGroup.group
                  .filter((msg) => msg.type === 'context')
                  .map((msg, index) => (
                    <div
                      key={index}
                      className="d-flex align-items-start mb-5 bg-textsite italic p-3"
                    >
                      <img
                        src={msg.content.sourceImage}
                        alt="Livre source"
                        className="max-w-40 mr-5"
                      />
                      <p className="text-white">{msg.content.pageContent}</p>
                    </div>
                  ))}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef}></div>
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
