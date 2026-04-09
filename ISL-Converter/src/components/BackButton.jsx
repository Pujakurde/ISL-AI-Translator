import { useNavigate } from 'react-router-dom'

function BackButton() {
  const navigate = useNavigate()

  return (
    <button className="back-btn" onClick={() => navigate(-1)}>
      ❮ Back
    </button>
  )
}

export default BackButton