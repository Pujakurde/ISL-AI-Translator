function BgCircles() {
  const circles = [
    { size: 90,  top: '5%',    left: '10%',   right: 'auto', bottom: 'auto' },
    { size: 70,  top: '10%',   right: '8%',   left: 'auto',  bottom: 'auto' },
    { size: 110, top: '25%',   left: '-30px', right: 'auto', bottom: 'auto' },
    { size: 80,  top: '20%',   right: '-25px',left: 'auto',  bottom: 'auto' },
    { size: 60,  top: '45%',   left: '5%',    right: 'auto', bottom: 'auto' },
    { size: 100,  top: '50%',   right: '-30px',left: 'auto',  bottom: 'auto' },
    { size: 70,  bottom: '30%',left: '15%',   right: 'auto', top: 'auto'    },
    { size: 90,  bottom: '20%',right: '10%',  left: 'auto',  top: 'auto'    },
    { size: 60,  bottom: '10%',left: '-20px', right: 'auto', top: 'auto'    },
    { size: 80,  bottom: '5%', right: '-20px',left: 'auto',  top: 'auto'    },
    { size: 50,  top: '70%',   left: '40%',   right: 'auto', bottom: 'auto' },
    { size: 65,  top: '35%',   left: '50%',   right: 'auto', bottom: 'auto' },
  ]

  return (
    <>
      {circles.map((c, i) => (
        <div
          key={i}
          className="bg-circle"
          style={{
            width: c.size,
            height: c.size,
            top: c.top,
            bottom: c.bottom,
            left: c.left,
            right: c.right,
          }}
        />
      ))}
    </>
  )
}

export default BgCircles