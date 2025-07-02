import './App.css';
import SmartNotebook from './demo';
import Mainpage from './mainpage';
import Mainpage2 from './mainpage2';
import { BrowserRouter as Router, Route, Routes, BrowserRouter } from 'react-router-dom';

function App() {
  return (
    // <>
    //   {/* <SmartNotebook /> */}
    //   <Mainpage/>
    // </>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={ <Mainpage/> } />
        <Route path = "/notebook" element ={ <SmartNotebook/> } />
        <Route path = "/mainpage2" element ={ <Mainpage2/> } />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
