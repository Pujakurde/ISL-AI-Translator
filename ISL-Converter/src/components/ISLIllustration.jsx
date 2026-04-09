import { useTheme } from '../ThemeContext'

function ISLIllustration() {
  const { isDark } = useTheme()

  return (
    <div className="illustration-wrap">
      <img
        src={isDark ? '/image_dark.png' : '/image_light.png'}
        alt="ISL Sign Language Character"
        className="isl-img"
      />
    </div>
  )
}

export default ISLIllustration