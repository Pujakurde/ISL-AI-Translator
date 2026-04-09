const team = [
  "Sarika Kore",
  "Kajal Kothalkar",
  "Pradumnya Kshirsagar",
  "Sayali Kurane",
  "Puja Kurde "
]

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">

        {/* LEFT — logo and desc */}
        <div className="footer-left">
          <span className="footer-logo">🤟 ISL Translator</span>
          <p className="footer-desc">♢ Bridging Communication with <br />  Indian Sign Language</p>
          <p className="footer-desc">B.Tech Final Year Project</p>
        </div>

        <div className="footer-links">

          {/* MIDDLE — features */}
          <div className="footer-col">
            <p className="footer-col-title">Features</p>
            <p className="footer-link">📷 Live Camera Detection</p>
            <p className="footer-link">🤖 AI ISL Assistant</p>
            <p className="footer-link">📖 ISL Dictionary</p>
            <p className="footer-link">🔄 Bidirectional Translation</p>
          </div>

          {/* RIGHT — team */}
          <div className="footer-col">
            <p className="footer-col-title">Our Team</p>
            {team.map((name, i) => (
              <p className="footer-link" key={i}>{name}</p>
            ))}
          </div>

        </div>
      </div>

      <div className="footer-bottom">
        <p className="footer-copy">© {new Date().getFullYear()} ISL Translator. All rights reserved.</p>
        <p className="footer-copy">Made with ❤️ for the deaf and hard of hearing community</p>
      </div>
    </footer>
  )
}

export default Footer